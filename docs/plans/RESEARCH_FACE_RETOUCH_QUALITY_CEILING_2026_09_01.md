# Research — Face Retouch Quality Ceiling

- **Date:** 2026-09-01
- **Status:** COMPLETE — full research and engineering specification only
- **Repository baseline:** `fafed66` (`fix(eyes): recalibrate eye-occlusion gate thresholds from study data`)
- **Scope:** still-image face retouching, shoot-level consistency, source fidelity,
  quality evidence, and optional learned assistance
- **Implementation state:** no engine, model, recipe, test, image, or output
  implementation is authorized or completed by this document

Related handoff documents:

- [Quality-ceiling implementation roadmap](PLAN_FACE_RETOUCH_QUALITY_CEILING_IMPLEMENTATION_2026_09_01.md)
- [Near-term AMG full-v2 engineering specification](PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md)
- [AMG batch review and research closeout](../review/REVIEW_AMGDAY32026_RETOUCH_BATCH_2026_09_01.md)
- [Face-operation visual audit](RESEARCH_FACEOP_VISUAL_AUDIT_2026_08_31.md)

---

## 1. Decision

The highest-quality face retouch Retouch can credibly build is **not one new model**.
It is a three-track system with strict boundaries:

1. **Photographic Truth Core — default**
   - deterministic or tightly bounded operations;
   - native-resolution source pixels remain the authority;
   - calibrated soft anatomy and uncertainty;
   - explicit mark policy;
   - material-aware skin, eye, lip, and teeth processing;
   - zero unauthorized geometry or identity synthesis;
   - exact-support compositing and complete evidence.
2. **Bounded Learned Assist — opt-in per operation**
   - learned denoise, deblur, blemish localization, or bilateral tone proposals;
   - residual, frequency, geometry, color, and mark budgets;
   - automatic bypass when evidence is weak;
   - source blend and deterministic rollback always available.
3. **Generative Studio — separate creative product mode**
   - diffusion/codebook/inpainting output is explicitly labeled synthetic;
   - never described as archival, identity-exact, or source-recovered;
   - never silently substituted for either of the first two tracks;
   - separate output naming, manifest fields, controls, and human acceptance.

The product thesis is therefore:

> Diagnose first, change the smallest justified support, preserve the subject's
> evidence, and prove every material edit.

“Highest quality” does not mean maximum smoothing, maximum whitening, the strongest
generative prior, or the highest generic image-quality score. It means the best
measured balance of:

- subject and likeness fidelity;
- correct anatomical localization;
- natural texture and material response;
- photographic color and tonal integrity;
- user-intended correction;
- consistency across a shoot;
- edit isolation and reversibility;
- robust delivery and review evidence.

This is a quality-ceiling plan. It does not claim that the current engine, any cited
paper, or the future system is globally state of the art.

---

## 2. Why the July frontier documents are no longer a current baseline

The July plans were useful, but several of their “missing” components now exist. A
new plan must use the current tree rather than repeat a stale backlog.

| Area | Current Retouch state | Remaining quality ceiling |
|---|---|---|
| Face parsing | BiSeNet 19-class ONNX plus MediaPipe-derived regions | Per-face paths still collapse logits through hard `argmax`; no calibrated posterior/entropy contract; no tongue class |
| Native detail | Full-quality face processing is native-resolution; three-band frequency separation and texture restoration exist | Every operation still needs measured band-specific retention and exact support |
| Skin appearance | Intrinsic decomposition, dichromatic specular work, chromophore v2, and compact makeup unmixing exist | These are separate approximations, not one calibrated skin-material inference system |
| Marks | Shared mark taxonomy and explicit `legacy`, `protect_identity`, and `preserve_all` policies exist | Detection confidence and cross-shot persistence are not yet a subject-level mark ledger |
| Lighting | Confidence-rated light direction is consumed by current processing; relight v2 works in an approximate linearized path | Geometry is a coarse MediaPipe-depth/Delaunay proxy, not a validated inverse-rendering face model |
| Harmony | Face/body texture, specular, banding, and mark-retention evidence exists | Most harmony metrics are observational; thresholds and automatic response remain uncalibrated |
| Learned restoration | NAFNet denoise infrastructure and model are present; Real-ESRGAN truthfully falls back when unavailable | Denoise runs as a broad pre-pipeline effect and lacks face-specific identity/mark/residual acceptance gates |
| Generative face work | Neural booster placeholders are deliberately disabled | Keep disabled in the default engine; any future synthesis belongs to Generative Studio |
| Certification | Render evidence and human-review contract v2 exist, including identity/likeness critical labels | The quality-ceiling corpus, per-operation evidence, calibrated gates, and final human results do not exist |

Two older statements require explicit correction:

- `retouch/lighting.py` is no longer an unused analyzer. Current
  `retouch/perf_optimizations.py` consumes a known light direction.
- harmony and class-aware mark policy are no longer unbuilt concepts. They are current
  modules, although their evidence and integration are not yet at the proposed ceiling.

The current implementation remains substantially stronger than the July baseline.
The new work begins at the gaps in the right-hand column, not from scratch.

---

## 3. Current code truth

### 3.1 Anatomy and masks

`retouch/parsing.py` currently provides:

- a 19-class 512×512 BiSeNet face parser;
- face-crop parsing for skin, lips, mouth interior, eyes, brows, neck, hair, and
  cloth;
- MediaPipe landmark-derived face oval, iris, sclera, under-eye, and other local
  regions;
- parser-failure fallback to landmark geometry;
- soft feathering after label-map construction;
- a full-frame hair path that retains the hair softmax probability.

The critical distinction is that a feathered hard label is **not** the same thing as a
calibrated soft semantic posterior. Both single-face and batch per-face paths use
`argmax` and nearest-neighbor label resizing before feathering. This discards:

- class ambiguity at boundaries;
- second-best class evidence;
- calibrated confidence;
- entropy or uncertainty;
- region-specific decision margins;
- evidence needed to distinguish “safe edit,” “reduced strength,” and “bypass.”

The full-frame hair path already demonstrates the desired direction: keep probability
and let a downstream consumer reduce influence where the model is uncertain.

The proposed system extends that principle to every critical facial region. It does
not merely blur a hard mask more attractively.

### 3.2 Precision and color representation

The per-face canvas is now predominantly `float32 [0,255]`, which avoids repeated
8-bit quantization during much of the face pipeline. This is a meaningful precision
improvement.

It is not yet an end-to-end linear-light material pipeline:

- many operations still work on encoded BGR or OpenCV Lab;
- `retouch/intrinsic.py` applies Rec.709 luma coefficients to normalized engine BGR
  without an explicit sRGB transfer-function decode;
- chromophore v2 correctly linearizes before optical density, but returns to encoded
  BGR for the wider pipeline;
- relight v2 uses a gamma 2.2 approximation inside its own bounded section;
- makeup unmixing uses a classical encoded-BGR alpha-composite approximation;
- some local feature paths still require `uint8` conversions.

The ceiling does not require converting every perceptual color operation to linear
light. It requires a declared representation per operation and prevents accidental
mixing:

- **linear scene/display-relative RGB** for illumination, reflectance, convolutional
  light transport, and physical compositing;
- **perceptual spaces** for color-distance decisions and intentionally perceptual
  controls;
- **encoded output RGB** only at declared I/O or model boundaries;
- no full-canvas color-space roundtrip when only a small support is being edited.

### 3.3 Skin texture and material

Current Retouch already has a serious classical texture stack:

- three-band frequency separation;
- adaptive high-band pore protection;
- directional wrinkle and local texture handling;
- pre/post smoothing micro-texture restoration;
- same-face texture transplant guarded by donor texture energy;
- intrinsic albedo×shading separation;
- dichromatic diffuse/specular approximation;
- relative melanin-like and hemoglobin-like optical-density coordinates;
- compact automatic makeup-artifact isolation;
- skin/face/body harmony observations.

The remaining limitation is coordination. Each module solves a local problem using a
different approximation and confidence model. The future engine needs a common
**face material evidence** object so that:

- skin smoothing does not flatten expression-dependent form;
- chromophore edits do not absorb lighting errors;
- specular correction does not erase intended makeup finish;
- makeup isolation does not spread a local estimate over bare skin;
- texture restoration does not reintroduce acne, sensor artifacts, or compressed
  noise;
- mark policy remains authoritative across all consumers;
- uncertainty reduces strength or bypasses instead of inventing a result.

Chromophore v2 is correctly documented as relative image coordinates, not literal
pigment concentration. That honesty remains a hard product boundary.

### 3.4 Geometry and lighting

Current relighting uses MediaPipe depth as a coarse surface proxy, Delaunay
interpolation, a low-order fitted shading field, and a bounded target light. This is
useful for subtle photographic shaping. It is not a high-fidelity facial geometry or
skin reflectance reconstruction.

The ceiling needs two levels of geometry:

1. **Fast operational geometry**
   - current landmarks and face mesh;
   - used for masks, crop scale, cautious local shading, and real-time behavior.
2. **Optional high-confidence geometry evidence**
   - a separately benchmarked parametric 3D face estimator;
   - used to improve normals, visibility, pose, and lighting diagnosis;
   - never allowed to replace source face texture in the default track;
   - bypassed when license, runtime, or confidence requirements fail.

The optional geometry model is an analyzer, not permission for face reshaping.

### 3.5 Marks and identity evidence

`retouch/marks.py` defines:

- mole;
- freckle;
- acne/blemish;
- scar;
- drawn makeup mark;
- stray hair;
- sensor dust;
- vellus sheen;
- unknown.

This is the correct semantic direction because “dark spot” is not an edit decision.
The same visual component could be a temporary blemish, a stable mole, makeup, or an
artifact. The explicit policy should decide preserve/remove/attenuate/enhance.

The current ceiling gap is temporal or shoot-level stability. A detector can miss a
mark in one frame. The new system should build a local-only, consented **subject mark
ledger** from multiple photographs when available:

- stable mark observed across frames → stronger preserve evidence;
- one-frame artifact aligned to sensor coordinates → possible sensor dust;
- movable occlusion or stray hair → frame-local evidence;
- unresolved class → preserve or bypass by policy;
- no automatic medical interpretation.

### 3.6 Learned restoration

Current `retouch/enhance.py` supports tiled NAFNet-like ONNX denoise and a classical
bilateral fallback. The installed NAFNet graph is CPU-pinned because the current
CoreML path was measured to be wrong and slower. Super-resolution truthfully reports
Lanczos when the verified Real-ESRGAN model is absent.

That runtime honesty is good. The quality ceiling needs additional content safety:

- characterize the degradation before enabling a learned restorer;
- analyze the proposed residual by semantic region and frequency band;
- block mark deletion and geometry drift;
- distinguish recoverable noise from real skin texture;
- require improvement over the classical fallback on the locked corpus;
- retain the source as an exact rollback;
- do not call a generative face prior “recovery” of unavailable detail.

### 3.7 Certification

The current v2 review contract already recognizes critical defects including:

- face swap;
- identity change;
- likeness change;
- geometry failure;
- unsafe mask;
- severe boundary bleed;
- severe color shift.

It also requires two distinct blinded reviewers and escalation to a third reviewer on
disagreement or a critical report. The ceiling extends this system; it does not replace
it with a single automatic “beauty score.”

---

## 4. What current primary research changes

This section maps the strongest useful research directions to Retouch. A cited method
is a benchmark candidate or design reference, not an adoption decision.

### 4.1 Face parsing: confidence and boundaries matter more than a model-name swap

[LaPa](https://ojs.aaai.org/index.php/AAAI/article/view/6832) contains more than
22,000 faces with 11 semantic categories and 106 landmarks. Its Boundary-Attention
Semantic Segmentation work is directly relevant to Retouch because our failures are
often boundary failures: lips versus mouth interior, eye versus skin, hair versus
skin, and face versus hand/occluder.

[FaRL](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_General_Facial_Representation_Learning_in_a_Visual-Linguistic_Manner_CVPR_2022_paper.html)
showed that a face-specific pretrained representation can improve both parsing and
alignment. It is a candidate representation benchmark, not a reason to add language
semantics to the production mask path.

[SegFace](https://github.com/Kartik-3004/SegFace) uses class-specific transformer
tokens and reports stronger long-tail face-region parsing on LaPa and CelebAMask-HQ,
including a mobile variant. This makes it the clearest modern parser challenger for an
offline bake-off against the current BiSeNet path.

Important adoption constraints:

- CelebAMask-HQ data and software carry non-commercial research restrictions even
  when a downstream repository's code license is permissive;
- a published mean F1 score does not prove our mouth, eye, cosplay, profile, hand,
  group, or tongue safety;
- the production choice must be based on Retouch's locked corpus, boundary metrics,
  calibration, runtime, model provenance, and license review;
- a new parser that is more accurate on average but worse on a critical class is not
  an upgrade.

**Research conclusion:** build a parser benchmark and posterior contract before
replacing BiSeNet. Soft calibrated evidence is P0; a model swap is conditional.

### 4.2 Face-retouch networks: local support and frequency reasoning are validated, but synthesis risk remains

[FFHQR / AutoRetouch](https://github.com/skylab-tech/ffhqr-dataset) established a
large professionally retouched face-pair resource based on FFHQ. It is useful for
style and aggregate retouch behavior, but aligned 1-megapixel face crops do not cover
Retouch's full camera-image, group, profile, costume, or delivery requirements.

[PPR10K](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html)
contains 11,161 high-quality RAW portrait photographs, three expert retouches, human
region masks, and group-level consistency. It strongly validates two product
requirements already relevant to Retouch:

- the human region deserves distinct treatment from the scene;
- a shoot must be evaluated for group-level tonal consistency, not only isolated
  frames.

[BPFRe](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Blemish-Aware_and_Progressive_Face_Retouching_With_Limited_Paired_Data_CVPR_2023_paper.html)
explicitly separates blemish localization from progressive repair and identifies the
core ambiguity between blemishes and identity features such as moles. This supports
Retouch's mark taxonomy and argues for a learned blemish proposal only behind an
explicit policy.

[RetouchFormer](https://ojs.aaai.org/index.php/AAAI/article/view/28404) formulates
retouching as soft inpainting with selective attention. It is relevant to difficult
large blemishes, but its content synthesis path belongs in a bounded benchmark until
mark and identity gates prove it safe.

[ICCV 2025 face retouching with spectral restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Face_Retouching_with_Diffusion_Data_Generation_and_Spectral_Restorement_ICCV_2025_paper.html)
introduces HGFR, a 25,000-pair, 1024×1024 benchmark, plus a soft mask branch,
frequency selection/restoration, and Laplacian multi-resolution fusion. The design is
highly relevant to Retouch's classical frequency stack.

Its evidence boundary matters:

- 23,760 of the pairs are synthetic, generated with fine-tuned diffusion models;
- the blemish prevalence statistic is based on a zero-shot CLIP classifier;
- synthetic diversity is valuable for stress tests but is not a substitute for
  identity-bearing real source/retouch pairs and professional human review.

[InstantRetouch (CVPR 2026)](https://openimaginglab.github.io/InstantRetouch/)
distills a diffusion teacher into one-step bilateral-space affine transforms. Its
deployment renderer changes tone and color while preserving high-frequency structure,
which is substantially safer for photographic retouch than unconstrained latent pixel
generation. The released code is Apache-2.0, but the full training/model/data chain
still requires provenance and license review.

This suggests an important future neural shape: **predict bounded photometric
operators, not replacement pixels**, whenever the user's intent can be represented by
exposure, contrast, color, or smooth local tone.

[Real-time high-resolution face retouching (2026)](https://doi.org/10.31881/TLR.2026.1826)
similarly predicts a low-resolution neutral-gray modulation layer and applies it to
the original high-resolution pixels through soft-light blending. Its evidence is less
comprehensive than the main conference work, but the architecture reinforces the same
principle: low-frequency learned decisions can be applied to authoritative
high-resolution source detail.

[BeautyGRPO (CVPR 2026)](https://beautygrpo.github.io/) uses a preference dataset and
reinforcement learning over a FLUX-based editor. The paper's five evaluation
dimensions—smoothing, blemish removal, skin tone/texture, personal-feature
preservation, and clarity—are useful additions to a human-review rubric.

BeautyGRPO is **not** a default-engine candidate:

- it is a generative FLUX editing path;
- its reward represents learned aesthetic preference;
- released project code and base-model weights have different licenses;
- preference alignment can encode style, cultural, age, gender-presentation, makeup,
  and dataset biases;
- a high identity-embedding score cannot prove pore, mole, scar, or wrinkle topology
  was preserved.

It belongs in Generative Studio research and, separately, its rubric may inform human
review.

**Research conclusion:** Retouch should learn from soft masks, multi-resolution
reasoning, bilateral operators, and preference rubrics. It should not collapse the
default photographic engine into an end-to-end beauty generator.

### 4.3 Inverse rendering: use stronger appearance evidence without surrendering pixels

[DECA](https://deca.is.tue.mpg.de/) estimates 3D shape, albedo, expression, pose,
illumination, and expression-dependent detail from a single image. It is useful as a
candidate source of normals, visibility, pose, and expression-aware detail evidence.
Its model/code license is for non-commercial scientific research, so it is a research
benchmark unless a deployable license is obtained.

[MICA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730249.pdf)
focuses on metrically accurate identity shape and reports large improvements on
metrical face reconstruction benchmarks. Its purpose is complementary to DECA:
identity shape rather than an editable photometric face. It can inform geometry
evaluation, but it is not required for subtle 2D retouch.

[A Morphable Face Albedo Model](https://openaccess.thecvf.com/content_CVPR_2020/html/Smith_A_Morphable_Face_Albedo_Model_CVPR_2020_paper.html)
separates diffuse and specular albedo using calibrated capture and explicitly builds
the model in linear sRGB. This reinforces the need to keep illumination and material
math in a declared linear representation.

[TRUST](https://trust.is.tue.mpg.de/) demonstrates that face-only albedo estimation is
ambiguous and that the wider scene carries lighting evidence. It also documents how
biased albedo priors can shift darker skin toward lighter estimates. Retouch should
therefore retain scene context for diagnosis even when the edit itself is face-local.

[HUST (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Ran_HUST_High-Fidelity_Unbiased_Skin_Tone_Estimation_via_Texture_Quantization_ICCV_2025_paper.html)
targets high-fidelity diffuse albedo and reports that three to six images improve and
stabilize multi-image estimates. The method is generative and should not directly
replace source pixels, but the multi-image result is strategically important for
Retouch's shoot workflow.

[Monocular Facial Appearance Capture in the Wild (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Monocular_Facial_Appearance_Capture_in_the_Wild_ICCV_2025_paper.html)
recovers geometry, diffuse albedo, specular intensity, and roughness from a short
head-rotation video while modeling visibility and unknown environment lighting.

Together, these works support a future **subject calibration** feature:

- use several frames or an optional short head-turn capture;
- estimate stable appearance, mark, texture, and lighting evidence;
- apply only bounded, frame-specific source-pixel edits in the default track;
- improve consistency without replacing the face with a reconstructed render.

This can become a real differentiator for event/cosplay batches because the current
input is already a shoot rather than a single anonymous crop.

### 4.4 Physics-guided blemish correction is a better default research path than unconstrained inpainting

[Controllable and Gradual Facial Blemishes Retouching via Physics-Based Modelling](https://arxiv.org/abs/2406.13227)
separates a texture layer from a diffuse layer and operates in melanin/hemoglobin-like
coordinates with a gradual control. This aligns closely with Retouch's existing
frequency, intrinsic, and chromophore components.

The useful design principle is not a claim that ordinary RGB reveals medical pigment
concentrations. It is:

- keep source high-frequency texture authoritative;
- correct lower-frequency blemish appearance in a constrained material space;
- expose gradual strength;
- preserve the ability to reconstruct and measure the residual;
- require camera/lighting confidence before interpreting color.

This is the preferred P1 research path for acne/pigmentation-like correction in the
Truth Core.

### 4.5 Restoration research confirms why one score is insufficient

[NAFNet](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)
and [Restormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)
are strong non-face-specific restoration architectures for denoise/deblur tasks. They
are better candidates for bounded sensor/degradation cleanup than face-generative
priors when source evidence is still present.

[CodeFormer](https://papers.neurips.cc/paper_files/paper/2022/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf)
uses a learned codebook prior and explicitly exposes a quality/fidelity tradeoff. It
is useful for severely degraded archival or web faces, but its generated atoms may
produce plausible rather than source-known detail.

The [NTIRE 2025 Real-World Face Restoration Challenge](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Chen_NTIRE_2025_Challenge_on_Real-World_Face_Restoration_Methods_and_Results_CVPRW_2025_paper.pdf)
first filters outputs through AdaFace identity similarity and then ranks valid entries
with multiple no-reference quality metrics and FID. It also reports that diffusion
priors are widely used to hallucinate high-frequency realism.

This is useful benchmark engineering, but it is insufficient for Retouch's default
contract:

- the identity thresholds vary by dataset;
- an embedding threshold is an alarm, not pixel or likeness proof;
- no-reference aesthetic/quality metrics can reward plausible synthetic texture;
- FID is a distribution statistic, not per-subject source fidelity;
- the restored “detail” may be invented.

The [perception-distortion tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)
formally explains why a more realistic-looking restoration can move farther from the
source signal. Retouch must therefore report fidelity and perceptual quality
separately.

**Research conclusion:** learned restoration may improve degraded inputs, but no
single perceptual, identity, or no-reference score can authorize it.

### 4.6 Identity embeddings are alarms, not completion evidence

[ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)
and [AdaFace](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)
provide useful face embeddings, with AdaFace explicitly addressing low-quality input.

For Retouch they may serve as:

- a before/after identity-drift alarm;
- a cross-shot association aid inside an access-controlled local session;
- one feature in a restoration benchmark;
- a trigger for mandatory human review.

They may not serve as:

- proof that likeness is unchanged;
- permission to reshape a face;
- proof that marks, pores, wrinkles, makeup, or age cues survive;
- an attractiveness or demographic classifier;
- a universal threshold copied from another dataset.

Identity evidence must combine embeddings, landmarks/geometry, stable marks,
operation support, source residuals, and blinded human review.

### 4.7 Skin-tone evaluation must not pretend a photograph is a medical phototype

Some older Retouch planning language treated image-derived ITA bands as equivalent to
Fitzpatrick skin types. The quality-ceiling plan rejects that shortcut.

Fitzpatrick type describes sun response and was not designed as a photographic skin
color scale. Recent controlled studies report poor agreement between image-extracted
ITA and colorimeter measurement under varying capture conditions, and warn against
mapping uncontrolled photographs to Fitzpatrick categories. See the
[prospective skin-tone scale comparison](https://www.nature.com/articles/s41746-025-02245-2)
and the [study of ITA under uncontrolled imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC13459198/).

Retouch may measure image-relative L*, chroma, or calibrated colorimetric values for
engineering analysis. It must not label a person's medical skin type from an ordinary
portrait.

For coverage and fairness evaluation, prefer:

- consented metadata when genuinely required;
- controlled chart/colorimeter measurements for a dedicated calibration subset;
- continuous, capture-aware image appearance measurements;
- per-condition error reporting rather than rigid human categories;
- explicit lighting/camera uncertainty;
- no race or ethnicity inference.

---

## 5. Quality contract

### 5.1 Seven non-negotiable pillars

#### Q1 — Subject fidelity

- no unrequested geometry change;
- no face swap or identity synthesis;
- stable marks follow explicit policy;
- expression-dependent lines are not automatically treated as defects;
- facial hair, makeup, scars, tattoos, and deliberate character marks survive unless
  the user asks otherwise;
- source and output remain directly comparable.

#### Q2 — Surgical localization

- every operation declares an effective support;
- every critical support derives from semantic, geometric, and confidence evidence;
- exact restoration outside effective support;
- occlusion and boundary uncertainty reduce effect or bypass;
- no full-canvas roundtrip for a local edit.

#### Q3 — Material realism

- preserve subject-specific pore and line statistics;
- separate low-frequency color/form from high-frequency texture;
- respect diffuse versus specular behavior;
- treat makeup as a distinct possible layer;
- do not invent physiologic certainty from camera RGB;
- avoid plastic smoothing, waxy specular, gray skin, or synthetic texture repetition.

#### Q4 — Optical facial-feature realism

- eyes retain lid shadow, scleral shading, iris texture, and catchlight consistency;
- lips retain vermilion boundary, natural texture, mouth/tongue separation, and
  physically plausible finish;
- teeth retain gum separation, inter-tooth shadow, translucency, and natural shade;
- no uniformly white sclera, fluorescent teeth, painted tongue, or glassy fake eyes.

#### Q5 — Photographic integrity

- declared ICC/color ingest and export;
- explicit encoded versus linear representation;
- native-resolution authoritative detail;
- no silent bit-depth, chroma, EXIF, or profile downgrade;
- reproducible runtime/model/provider evidence;
- group-level tone consistency without forcing identical faces.

#### Q6 — Explainability and reversibility

- analyzer evidence is distinct from edit decisions;
- every automatic decision has a reason and confidence;
- every optional model has a version, hash, license/provenance record, and fallback;
- source assets remain unchanged;
- every learned or classical stage can be disabled and compared independently;
- generative output is visibly and structurally separated.

#### Q7 — Evidence-backed release

- automatic gates are calibrated on a locked corpus;
- tests are mutation-sensitive;
- operation isolation is measured on real images;
- source/output A/B is blinded;
- two reviewers plus a third on disagreement/critical defect;
- no “highest quality” claim from a contact sheet, a model paper, or a focused test
  alone.

### 5.2 Non-goals

The default face engine will not:

- infer attractiveness;
- infer race, ethnicity, health, or medical skin state;
- automatically feminize, masculinize, age, or de-age a subject;
- reshape facial geometry to a learned beauty prior;
- erase every pore, wrinkle, freckle, scar, or mole;
- generate catchlights, pores, teeth, eyelashes, or skin unless an explicitly separate
  creative feature is chosen;
- call plausible generated detail “recovered original detail”;
- optimize a person toward a population-average face;
- use public benchmark licenses as assumed commercial authorization.

---

## 6. Proposed quality-ceiling architecture

```text
source asset
  -> decode/orientation/ICC/metadata truth
  -> detection + multi-scale face evidence
  -> soft region posteriors + landmarks + visibility + uncertainty
  -> face diagnosis (material, marks, lighting, degradation, occlusion)
  -> explicit user/recipe intent and operation budgets
  -> deterministic Photographic Truth Core
  -> optional Bounded Learned Assist per approved operation
  -> source-authoritative exact-support composite
  -> automatic fidelity/material/color/isolation gates
  -> backoff or bypass on failed evidence
  -> blinded human review candidates
  -> color-managed export + complete manifest
```

The system separates four concepts that are too easily conflated:

| Concept | Meaning | May alter pixels? |
|---|---|---:|
| Observation | Raw detector/model/metric output | No |
| Diagnosis | Interpreted condition with confidence and limitations | No |
| Intent | User/recipe-approved goal and strength budget | No |
| Operation | A bounded pixel or geometry transform | Yes, only inside declared support |

An analyzer can say “possible local redness under mixed-light uncertainty.” It cannot
silently decide “remove redness.” The intent and policy layer owns that decision.

### 6.1 Stage A — source truth

Required inputs to all later evidence:

- source artifact hash;
- decode library/version;
- original dimensions, orientation, channels, bit depth, ICC state, and metadata
  policy;
- normalized working-space declaration;
- RAW development settings when applicable;
- camera/display transform confidence;
- no implicit profile relabeling.

If color is untrusted, color-sensitive material diagnosis must reduce confidence or
bypass. It may not present precise pigment-like measurements as fact.

### 6.2 Stage B — face observation graph

Each detected face receives a stable observation ID for the render and, where a
shoot-level association is authorized, a session-local subject ID.

The evidence graph should contain:

- detector boxes and confidence;
- 2D landmarks and visibility;
- optional 3D pose/normals evidence;
- per-class semantic posterior maps;
- boundary confidence;
- posterior entropy/uncertainty;
- parser arm and model hash;
- person/hair/hand/occluder evidence;
- eye openness/occlusion evidence;
- mouth state and mouth-safety decision;
- face scale and usable native pixels;
- evidence validity reasons.

No downstream module should have to guess whether a mask came from BiSeNet,
landmarks, a fused path, or fallback.

### 6.3 Stage C — diagnosis graph

The diagnosis stage emits evidence, not edits:

- noise, blur, compression, clipping, and demosaic artifacts;
- low/mid/high-band skin statistics;
- local blotch and color residuals;
- relative chromophore coordinates and confidence;
- intrinsic shading/albedo reconstruction confidence;
- diffuse/specular likelihood and roughness proxy;
- makeup coverage/finish likelihood;
- mark records and stability;
- lighting direction and uncertainty;
- face/body/shoot harmony observations;
- unsafe or unmeasurable regions.

Every diagnosis includes:

- method/version;
- source inputs;
- confidence;
- known ambiguity;
- unit/scale;
- whether it is single-frame or multi-frame evidence;
- allowed consumers.

### 6.4 Stage D — operation plan

The operation plan is a deterministic contract assembled from:

- chosen recipe;
- explicit user controls;
- mark policy;
- diagnosis confidence;
- face scale;
- region safety;
- per-operation maximum residual budgets;
- conflicts and precedence.

Example precedence rules:

1. Preserve masks dominate heal/attenuate masks.
2. Unsafe eye or mouth state bypasses cosmetic operations on that region.
3. User-specified mark actions dominate automatic class suggestions.
4. Color correction cannot override an invalid color context.
5. Learned assist cannot expand deterministic support.
6. Geometry stays disabled unless explicitly requested.
7. Generative Studio cannot be entered through a fallback.

### 6.5 Stage E — source-authoritative render

Every operation must return enough evidence to reconstruct its impact:

| Field | Contract |
|---|---|
| Operation ID/version | Stable named implementation |
| Input artifact/stage hash | Exact upstream state |
| Proposed result | Same dimensions unless the operation explicitly owns resize/export |
| Effective support | Final alpha/support after every safety exclusion |
| Residual | Proposed minus input in the declared representation |
| Strength/budget | Requested, resolved, and applied values |
| Evidence inputs | Region/diagnosis IDs and confidence |
| Fallback/bypass | Explicit state and reason |
| Runtime/model | Provider, model hash, and provenance when learned |
| Metrics | In-support change, outside-support change, color/texture/geometry deltas |

The compositor applies the final support and restores the exact upstream pixels
outside it. An operator's internal full-frame conversion must not leak into delivery.

### 6.6 Stage F — automatic backoff

Backoff is allowed only for calibrated, monotonic controls. It should proceed from the
user-approved target toward the source:

1. render proposed strength;
2. evaluate hard and calibrated gates;
3. if a recoverable gate fails, reduce the relevant operation only;
4. rerender from the same upstream input, not from the failed output;
5. stop at the strongest passing value;
6. if no nonzero value passes, bypass;
7. record every attempt and reason.

There is no backoff from a generative result into the Truth Core. They are separate
render paths.

---

## 7. Engineering research packages

### P0.1 — calibrated soft face-region evidence

**Problem:** per-face parser logits are collapsed to hard labels, so downstream code
cannot distinguish confident interior pixels from ambiguous anatomy.

**Deliverable:** one region-evidence contract containing per-class posterior,
calibrated confidence, entropy, boundary confidence, source arm, and validity.

**Candidate benchmark arms:**

- current BiSeNet logits retained and calibrated;
- current logits plus landmark/visibility fusion;
- SegFace challenger;
- FaRL-based parsing challenger if runtime/license permits;
- landmark-only fallback retained as a separate arm, never treated as equivalent.

**Required classes/derived regions:**

- skin;
- upper/lower lip;
- mouth interior;
- teeth;
- tongue or explicit unknown-mouth tissue;
- sclera, iris, eyelids, brows;
- nose;
- hair;
- neck;
- facial hair;
- eyeglasses/accessories;
- hand/foreground occluder;
- unknown/unsafe.

No public parser is assumed to provide all classes. Derived critical masks may combine
posterior, geometry, color/material evidence, and dedicated calibration.

**Hard requirements:**

- strength-zero path is byte-identical;
- no NaN/Inf posterior;
- posterior class axis and normalization validated;
- critical mask uncertainty is available to consumers;
- model/fallback arm recorded;
- invalid evidence bypasses rather than substituting zeros and claiming confidence;
- tongue/open-mouth behavior remains subject to the dedicated AMG calibration plan.

**Completion boundary:** not complete when a new model runs. Complete only when the
locked corpus shows better critical-class boundary performance/calibration without
runtime, license, or safety regression, and mutations prove consumers use the new
uncertainty.

### P0.2 — exact-support operation protocol

**Problem:** local operations can alter pixels outside their conceptual mask through
full-canvas color conversion, dtype conversion, or ROI recomposition.

**Deliverable:** a common operation-result and composite protocol applied first to
lips/teeth, then every face operation.

**Exact invariants:**

- zero effective support → output byte-identical to operation input;
- outside effective support → output byte-identical to operation input;
- strength zero → output object/value identity according to the declared public
  contract;
- no operation may silently enlarge support after policy resolution;
- support visualization and residual heatmap available for QA;
- removing final source restoration must fail mutation tests.

**Completion boundary:** every migrated operation passes unit, mutation, real-image
isolation, multi-face, edge-of-frame, and color-space roundtrip tests. “Mean leakage is
small” is not completion where exact restoration is feasible.

### P0.3 — representation ledger and color-safe material math

**Problem:** float precision is improved, but encoded, perceptual, and approximate
linear operations remain interleaved.

**Deliverable:** a representation declaration for every face operation plus shared
conversion primitives.

**Named representations:**

- `encoded_srgb_f32` or the actual canonical encoded working profile;
- `linear_working_rgb_f32`;
- `cie_lab_f32` with white point/profile declared;
- `oklab_f32`;
- `optical_density_relative_f32`;
- mask/alpha with explicit normalization;
- output integer encoding only at I/O boundaries.

**Migration order:**

1. shared conversion tests and reconstruction closure;
2. intrinsic decomposition;
3. relight/specular;
4. chromophore and makeup coordination;
5. frequency convolution audit;
6. local feature conversions;
7. end-to-end color and banding evidence.

**Non-goal:** force Lab or other intentionally perceptual decisions into linear RGB.
The goal is correct and explicit math, not one universal color space.

### P0.4 — subject mark ledger

**Problem:** per-frame classification can confuse stable identity marks, temporary
blemishes, makeup, hair, and sensor artifacts.

**Deliverable:** optional shoot-level, local-only mark evidence.

**Ledger rules:**

- coordinate observations through face geometry without persisting biometric
  identity beyond the authorized job;
- retain per-frame observations and uncertainty;
- stable cross-frame marks increase preserve confidence;
- unresolved marks default to the explicit policy, normally preserve;
- no disease or health label;
- no remote service required;
- ledger destroyed or retained according to the job's declared privacy policy.

**Completion boundary:** cross-shot mark retention improves on the locked corpus,
false-preserve and false-remove costs are documented separately, and users can inspect
or override every destructive action.

### P1.1 — unified face material evidence

**Problem:** current intrinsic, chromophore, specular, makeup, and texture modules do
not share one confidence or reconstruction contract.

**Deliverable:** a source-reconstructing evidence stack:

```text
observed face appearance
  ~= illumination/visibility
   x diffuse reflectance
   + specular response
   + makeup layer contribution
   + source texture/detail residual
   + sensor/compression residual
```

This is an engineering model, not a claim of unique physical recovery from one RGB
image.

**Outputs:**

- coarse geometry/normals and confidence;
- diffuse/albedo-like estimate;
- shading estimate;
- specular intensity and roughness proxy;
- relative chromophore coordinates;
- makeup alpha/finish likelihood;
- texture bands and orientation statistics;
- degradation residual;
- reconstruction error;
- ambiguity flags.

**Editing rule:** default operations modify the smallest interpretable component and
recompose with authoritative source detail. When decomposition confidence is low, use
the existing conservative classical operation or bypass.

**Candidate research arms:**

- current classical modules coordinated through a common contract;
- current modules plus calibrated scene-light evidence;
- optional DECA-like normals/visibility evidence;
- optional multi-frame appearance evidence;
- no generative albedo pixels in default output.

**Completion boundary:** the coordinated model must reduce real visual defects and
component cross-talk versus current Retouch while maintaining reconstruction,
identity, mark, texture, color, and runtime gates.

### P1.2 — material-aware skin correction

**Priority corrections:**

1. random low/mid-frequency blotch without pore removal;
2. local redness/chroma variance with neutral shading preserved;
3. specular hotspot control with a residual natural finish;
4. makeup coverage evening only where makeup evidence exists;
5. directional wrinkle/chapping attenuation with retention floors;
6. adaptive micro-texture restoration from the same face/source;
7. face/body/shoot material parity.

**Never default:**

- pore synthesis;
- freckle redistribution;
- global foundation synthesis;
- wrinkle erasure;
- skin-lightening target;
- population-average “healthy” color target.

**Control model:** every correction is gradual and target-seeking relative to the
subject/source, not a universal target appearance.

### P1.3 — optical eye, lip, and teeth models

The highest perceived quality does not come from skin alone. Small errors in eyes,
mouth, and teeth dominate viewer trust.

#### Eyes

- preserve or restore existing catchlight evidence consistently between eyes;
- retain lid/corneal/scleral shading after whitening;
- apply relative vascular/chroma correction rather than flat white paint;
- enhance real iris radial texture without generating a new iris;
- gate every operation by eye visibility and occlusion;
- keep catchlight synthesis and gaze correction in separate creative controls.

#### Lips

- tongue-safe and mouth-safe support from the AMG plan;
- upper/lower lip evidence where available;
- diffuse/specular finish control rather than generic bright-spot boost;
- preserve vermilion boundary and natural vertical texture;
- target chapping/flake evidence without flattening the entire lip;
- keep plumping or geometry change explicit and off by default.

#### Teeth

- tooth/gum/lip-interior/shadow separation;
- natural target range rather than maximum white;
- preserve inter-tooth shadow, enamel specular, and incisal translucency;
- correct per-tooth outliers only with sufficient segmentation evidence;
- bypass on braces, gems, grills, severe occlusion, or uncertainty unless a dedicated
  calibrated arm exists.

**Completion boundary:** each region ships independently only after its own corpus,
isolation, mutation, material, color, and blinded visual acceptance. A skin upgrade
does not certify eyes, lips, or teeth.

### P1.4 — shoot-level subject calibration

**Problem:** single-image inference confuses stable subject appearance with frame
lighting, pose, focus, compression, and temporary occlusion.

**Deliverable:** an optional preflight over a folder or short capture that generates
reviewable, local-only subject evidence.

**Candidate stable evidence:**

- mark inventory;
- baseline texture energy by facial zone;
- relative diffuse color tendency;
- common makeup finish;
- facial-hair regions;
- geometry/pose template;
- per-shot lighting deviation;
- frame-specific anomaly versus shoot consensus.

**Use cases:**

- prevent a mole from disappearing in one frame;
- keep skin finish consistent despite exposure variation;
- distinguish sensor dust from a face mark;
- avoid over-smoothing the sharp frame to match a soft frame;
- choose the best evidence frame for diagnosis without copying its pixels blindly;
- flag inconsistent retouch intensity before delivery.

**Privacy boundary:** this is not a people database. Association is job-local,
purpose-limited, access-controlled, and deleted/retained by explicit policy.

**Completion boundary:** shoot-level evidence improves consistency without increasing
per-frame identity/mark/color failures, and single-image mode remains fully supported.

### P2.1 — bounded learned restoration laboratory

**Goal:** determine whether learned assistance can beat current classical operations
without losing source truth.

**Initial candidate arms:**

- current NAFNet denoise;
- current bilateral fallback;
- Restormer or another licensed non-generative restoration challenger;
- a low-frequency neutral-modulation or bilateral-grid model;
- a blemish-localization network whose output is a proposal mask only;
- CodeFormer/NTIRE-style generative restoration as a separate research control, not a
  default candidate.

**Every learned result is decomposed into:**

- low-frequency luminance residual;
- chroma residual;
- mid-band structure residual;
- high-frequency texture residual;
- landmark/edge displacement;
- mark-region residual;
- outside-support residual;
- identity-embedding change;
- perceptual metric change;
- runtime/provider/model evidence.

**Hard gates:**

- no dimension change;
- no support expansion;
- no non-finite output;
- no unauthorized geometry;
- no critical mark deletion;
- no silent fallback/model substitution;
- deterministic seed/runtime behavior where the operation is declared deterministic;
- exact upstream rollback.

**Calibrated gates:**

- acceptable residual per frequency band;
- improvement in degradation estimate;
- acceptable color drift;
- texture non-inferiority;
- identity and likeness alarm thresholds;
- human preference and defect rate.

No external paper threshold is copied into production. All non-exact thresholds are
calibrated against the locked Retouch corpus.

**Completion boundary:** a learned arm enters Bounded Learned Assist only when it
beats the classical baseline on its declared task, passes every hard gate, shows no
important condition-specific regression, has approved model/data licensing, and is
approved in blinded human review.

### P2.2 — learned planning without learned identity ideals

Recent instruction-retouch and preference research suggests a safer use of large
models: propose operations and settings, then execute transparent tools.

Retouch may research a planner that converts intent such as “clean but natural event
portrait” into:

- named existing operations;
- bounded slider proposals;
- reason/confidence;
- which regions will be affected;
- which evidence caused the proposal;
- which decisions need human confirmation.

The planner must not:

- directly emit final face pixels in the Truth Core;
- infer attractiveness or a demographic beauty norm;
- decide stable marks are defects;
- bypass region safety;
- enter Generative Studio silently;
- certify its own output.

User preference should be learned from explicit A/B choices and named styles where
possible, not inferred from the person's face.

### P3 — Generative Studio

Generative Studio exists for requests where source evidence cannot satisfy the goal:

- severe restoration with missing detail;
- creative makeup or freckles;
- catchlight synthesis;
- expression, gaze, or geometry alteration;
- large blemish inpainting where no source texture exists;
- deliberate editorial idealization.

Required separation:

- separate UI mode and warning;
- separate recipe namespace;
- separate artifact suffix;
- `generative=true` and model/seed/hash in the manifest;
- original retained beside output;
- no “recovered” or “identity-exact” wording;
- mandatory human review for delivery presets;
- no use as an automatic fallback;
- license/content-provenance review.

The research question is not “can it look good?” It is “can users understand exactly
when source evidence ended and synthesis began?”

---

## 8. Corpus specification

### 8.1 Corpus layers

The quality ceiling needs four distinct corpora. They may share assets only when the
license, consent, and split rules permit.

#### C1 — synthetic unit corpus

Purpose:

- exact mask and residual tests;
- color and representation closure;
- known noise/blur/compression;
- controlled marks and boundaries;
- mutation testing;
- deterministic extreme conditions.

It cannot certify natural appearance or likeness.

#### C2 — licensed real single-image corpus

Purpose:

- real pores, makeup, hair, facial hair, specular, skin color, camera noise, and
  occlusion;
- critical region masks;
- source/output human review;
- per-operation isolation.

Requirements:

- explicit training/evaluation/commercial rights as applicable;
- adult subjects by default; any minor data requires a separate documented legal and
  consent path;
- identity-disjoint train/calibration/test;
- originals, not social-media recompressions alone;
- no identity details committed to the public repository.

#### C3 — professional paired-retouch corpus

Purpose:

- expert correction targets;
- operation intent and masks;
- style variability;
- human preference calibration;
- non-generative high-end reference.

For a useful subset, retain:

- source RAW/developed image;
- final expert retouch;
- layer or operation grouping when the retoucher can provide it;
- intended preserve/remove mark annotations;
- retoucher confidence and notes;
- more than one expert treatment for ambiguous style cases.

There is no single “ground truth beauty” image. Multiple expert retouches should be
treated as valid style samples, not averaged into one face.

#### C4 — shoot and multi-view corpus

Purpose:

- job-local subject calibration;
- mark stability;
- group-level color/material consistency;
- pose/lighting/occlusion variation;
- same-subject leakage prevention;
- batch-level acceptance.

Include three to six usable portraits for part of the corpus and optional short
head-turn sequences for the appearance-capture research arm.

### 8.2 Required observable conditions

Coverage must be tracked at least across:

| Axis | Required conditions |
|---|---|
| Face size | tiny/group face, medium, high-resolution close-up |
| Pose | frontal, three-quarter, profile, up/down tilt |
| Expression | neutral, smile, laugh/open mouth, squint, closed eye |
| Lighting | soft, hard, mixed color, backlit, flash, clipped highlights, deep shadow |
| Capture | RAW, camera JPEG, phone, high ISO, compression, slight blur, sharp studio |
| Skin appearance | continuous calibrated/relative appearance coverage; no automatic race label |
| Texture | low-visible pores, high-visible pores, fine lines, mature texture, facial hair, vellus sheen |
| Marks | acne-like blemish, mole, freckles, scar, tattoo, drawn/cosplay mark, sensor dust |
| Makeup | none, light, foundation, matte, dewy, strong color, glitter/cosplay |
| Occlusion | glasses, hair/wig, hand, prop, mask edge, hat shadow |
| Mouth | closed, teeth, braces/accessory, lipstick, open mouth, visible tongue |
| Eyes | open, squint, closed, hair-covered, glasses reflection, asymmetric visibility |
| Composition | one face, unequal multi-face, crowd, poster/printed-face distractor |
| Body context | face only, neck/shoulders, exposed arms/hands, colored costume/wig |

### 8.3 Annotation levels

Not every image requires every annotation. Use nested levels:

1. **Job/source truth:** hashes, rights, capture, profile, metadata policy.
2. **Face truth:** boxes, landmarks/visibility, pose, face size, subject-local group.
3. **Critical semantics:** eyes, sclera, iris, lips, mouth interior, teeth, tongue,
   hair, hands/occluders.
4. **Mark truth:** location, explicit keep/remove/uncertain intent, stability across
   frames.
5. **Material truth:** controlled subsets with diffuse/specular/color references or
   multi-view evidence.
6. **Retouch truth:** operation intent, support, expert result, ambiguity.
7. **Review truth:** blinded decisions and structured defects.

### 8.4 Split discipline

Minimum split rules:

- no identity overlap between train, calibration, and sealed test;
- no burst/shoot leakage across splits;
- hold out cameras or capture pipelines for a generalization slice;
- hold out at least one retoucher/style in the paired corpus;
- keep synthetic families/prompts disjoint where synthetic data is used;
- keep the sealed test inaccessible during threshold tuning;
- version every corpus and annotation revision.

### 8.5 Fairness and privacy

- no race/ethnicity inference;
- no uncontrolled-photo Fitzpatrick labeling;
- report performance over controlled/relative appearance and capture conditions;
- obtain consent and purpose-limit biometric association;
- encrypt/access-control private source assets and ledgers;
- commit only opaque IDs, schemas, and aggregate evidence;
- document under-represented conditions rather than claiming universal coverage;
- never call an unmeasured group “passed.”

---

## 9. Evaluation system

### 9.1 Metrics are a vector, not a single quality score

The release record should preserve at least these metric families separately:

| Family | Examples | Meaning |
|---|---|---|
| Support/isolation | outside-support max/mean residual, changed-pixel count | Did the operation touch only authorized pixels? |
| Geometry | landmarks, contour displacement, face-shape residual | Did unrequested structure move? |
| Identity alarms | ArcFace/AdaFace similarity, mark retention | Is there evidence of likeness or stable-feature drift? |
| Pixel fidelity | L1/L2, PSNR, SSIM in relevant support | How far did output move from source/reference? |
| Perceptual fidelity | LPIPS, DISTS | Did learned perceptual/texture structure move? |
| Texture | per-band energy, radial/orientation spectrum, repetition | Were pores/lines retained without synthetic pattern? |
| Color | ΔE, neutral drift, skin/body/shoot relative color | Did intended color change without collateral shift? |
| Material | reconstruction error, diffuse/specular/roughness proxy deltas | Did component edits remain coherent? |
| Degradation | noise/blur/compression estimate before/after | Did restoration solve its declared task? |
| Region safety | eye/mouth/tongue/teeth/occluder mutation | Were critical anatomical regions protected? |
| Human | blinded preference, defects, confidence, disagreement | Does it actually look better and remain the subject? |

No metric may silently substitute for another. A high SSIM does not prove a good
retouch; a high preference score does not prove fidelity; an identity embedding does
not prove stable marks; a pore-energy ratio does not prove natural pore topology.

### 9.2 Exact gates

These do not require corpus-derived thresholds:

- zero-strength identity;
- zero-support identity;
- exact pixels outside effective support where the operation contract permits exact
  restoration;
- no dimensions/channels/dtype change unless declared;
- finite arrays and valid normalized masks;
- no unauthorized geometry path;
- no unrecorded model or fallback;
- no generative result in the Truth Core;
- source file unchanged;
- manifest and artifact hashes complete;
- deterministic rerun equality for deterministic paths.

### 9.3 Calibrated gates

These require the locked corpus and must not be invented in documentation:

- parser posterior temperature and confidence bins;
- semantic/boundary acceptance by class and face size;
- eye/mouth/tongue safety thresholds;
- mark classifier thresholds;
- texture retention bands;
- color/material drift budgets beyond existing operation-specific caps;
- geometry and identity alarm thresholds;
- degradation improvement floor;
- learned residual budgets;
- shoot-consistency tolerances;
- automatic backoff decision points;
- human preference non-inferiority/superiority targets.

Threshold reports must show distributions, failure examples, and condition-specific
performance—not only an aggregate optimum.

### 9.4 Parser evaluation

Required metrics:

- per-class IoU/F1;
- boundary F-score at face-size-normalized tolerances;
- expected calibration error or equivalent reliability analysis;
- critical-region false-positive and false-negative cost;
- posterior entropy versus actual error;
- face-size, pose, makeup, occlusion, and parser-arm slices;
- runtime and memory at native workflow scale;
- model availability/provider/fallback truth.

For lips, teeth, tongue, eyes, and hands, report cost-weighted failure separately from
mean F1.

### 9.5 Operation isolation

Every operation is tested in isolation on:

- synthetic exact masks;
- real single faces;
- multi-face images;
- edge-of-frame faces;
- profile/occlusion cases;
- parser and fallback arms;
- zero, low, nominal, and maximum strength;
- repeated application where idempotence or bounded accumulation is expected;
- color/profile variants;
- mutation removing the final source restore or safety exclusion.

Artifacts:

- source;
- isolated output;
- absolute residual;
- support overlay;
- outside-support residual summary;
- per-band residual;
- operation evidence JSON.

### 9.6 Identity and likeness review

Automatic alarms:

- embedding similarity from at least one calibrated recognizer;
- landmark/contour drift;
- stable mark survival;
- facial-hair/makeup region survival;
- local high-frequency correspondence;
- unrequested age/expression cue change flags where measurable.

Human reviewers answer separately:

1. Is this clearly the same subject and likeness?
2. Did any face shape, expression, age cue, or distinctive feature change without
   intent?
3. Were moles, freckles, scars, makeup marks, facial hair, and wrinkles handled
   according to policy?
4. Does any region look synthesized, repainted, or copied?

Any critical identity/likeness finding rejects the candidate regardless of aggregate
preference.

### 9.7 Material and texture review

At 100% and normal viewing size, inspect:

- pores and fine lines;
- blotch versus form separation;
- specular shape and residual natural finish;
- makeup boundaries and finish;
- vellus hair and facial hair;
- repeated/transplanted texture;
- haloing at lips, nose, brows, hairline, and jaw;
- face/body texture and color parity;
- plastic, waxy, gray, crunchy, or over-sharpened appearance.

### 9.8 Human acceptance

Use the existing review-v2 mechanics:

- blinded source/candidate side assignment;
- two independent reviewers;
- third reviewer on disagreement or any critical defect;
- structured region/severity labels;
- reviewer confidence;
- no source/output naming leaks;
- immutable pair commitments;
- review records separate from render evidence.

Add quality-ceiling dimensions:

- correction effectiveness;
- natural skin/material appearance;
- personal-feature preservation;
- eye/mouth/teeth realism;
- photographic color and clarity;
- face/body/shoot consistency;
- overall preference.

Reviewers must be able to approve “less changed” over “more flawless.”

### 9.9 Statistical completion

Before implementation, perform a power analysis from pilot variance and the minimum
meaningful preference/defect difference. Do not pick a sample size because it is round
or convenient.

Release analysis must include:

- confidence intervals;
- paired comparisons against the current best Retouch baseline;
- condition-specific defect rates;
- non-inferiority on all fidelity/safety pillars;
- sensitivity analysis for uncertain reviews;
- documented missing/underpowered conditions;
- no tuning on sealed test outcomes.

---

## 10. Mutation program

The suite is not trusted until intentional defects fail named tests.

Required mutations include:

### Masks and safety

- replace posterior with hard `argmax`;
- swap left/right eye evidence;
- remove eye-visibility gate;
- ignore parser uncertainty;
- remove tongue exclusion;
- use outer lip polygon as full lip support;
- remove hand/occluder exclusion;
- force fallback arm to report model confidence;
- replace invalid evidence with zeros marked valid.

### Support and color

- remove exact outside-support restoration;
- convert the full canvas through Lab for a local operation;
- quantize the per-face canvas to `uint8` mid-pipeline;
- apply linear-light math to encoded RGB;
- relabel a profile without conversion;
- remove ICC/export validation.

### Marks and identity

- classify all dark spots as blemishes;
- ignore preserve-mask precedence;
- delete stable mark ledger evidence;
- accept identity embedding alone;
- enable geometry without explicit intent;
- allow learned support expansion.

### Learned models

- substitute a different model hash;
- change provider silently;
- return fallback while reporting neural success;
- inject high-frequency synthetic texture;
- return non-deterministic output from a deterministic mode;
- skip residual analysis;
- cross the Generative Studio boundary through an error fallback.

### Certification

- duplicate reviewer IDs;
- reveal source/output side;
- accept one reviewer;
- ignore a critical defect;
- omit a corpus condition or artifact hash;
- certify on warning-only automatic QA.

Each mutation needs at least one direct failing assertion and, for high-risk paths, a
real-image evidence artifact.

---

## 11. Prioritized implementation sequence

### Phase 0 — finish the already-proven mechanical blockers

**Scope:** the companion AMG full-v2 engineering plan.

1. exact-support lip/teeth/Lab roundtrip containment;
2. canonical color-managed ICC/EXIF/4:4:4 export;
3. complete batch manifest and atomic/resumable runner;
4. tongue/open-mouth calibration and fail-open safety;
5. mutation-sensitive isolation and delivery tests;
6. rerender and human A/B.

**Why first:** a sophisticated material or neural model cannot compensate for leaked
roundtrips, incomplete evidence, unsafe mouth masks, or uncertain exports.

**Completion boundary:** exactly the one defined in the
[AMG full-v2 engineering specification](PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md).

### Phase 1 — evidence foundation

1. operation-result/exact-support protocol;
2. representation ledger and shared conversion tests;
3. soft posterior extraction from current BiSeNet;
4. posterior calibration and critical boundary corpus;
5. common face observation/diagnosis graph;
6. per-operation residual artifacts;
7. parser challenger benchmark.

**Release effect:** no automatic visible change is required. This phase makes future
changes measurable and safe.

**Completion boundary:** current output remains compatible where intended; new
evidence is complete, validated, mutation-sensitive, and consumed by at least one
critical safety path.

### Phase 2 — identity and material foundation

1. subject mark ledger;
2. unified material evidence contract;
3. intrinsic/linear-light correction spike;
4. coordinated chromophore/specular/makeup confidence;
5. scene-aware lighting evidence;
6. texture and reconstruction metrics;
7. calibrated automatic backoff.

**Release effect:** begin with diagnostic-only evidence. Enable visible operations one
at a time after isolation and human approval.

**Completion boundary:** each enabled operation passes exact support, component
reconstruction, mark/identity, texture, color, and visual gates independently.

### Phase 3 — optical feature quality

1. teeth/gum/mouth semantic refinement;
2. natural teeth material correction;
3. lip diffuse/specular/texture correction;
4. eye scleral/corneal/iris material correction;
5. face/body/shoot harmony calibration;
6. independent region certification.

**Completion boundary:** no shared “face passed” shortcut. Eyes, lips, teeth, and skin
each have their own certified corpus and completion record.

### Phase 4 — shoot-level intelligence

1. session-local face association;
2. multi-frame mark stability;
3. multi-frame texture/material baseline;
4. shot-specific deviation diagnosis;
5. group-level retouch consistency plan;
6. shoot-level review contact sheet and outlier detection;
7. privacy retention/destruction policy.

**Completion boundary:** measurable consistency improvement over independent-frame
processing with no fidelity or privacy regression.

### Phase 5 — bounded learned assists

1. locked degradation and blemish benchmark;
2. current NAFNet versus classical baseline audit;
3. licensed restoration challengers;
4. low-frequency/bilateral operator model spike;
5. learned blemish proposal-mask spike;
6. residual, mark, geometry, and color guards;
7. failure injection and model/provider provenance;
8. blinded acceptance.

**Completion boundary:** only winning task-specific arms ship, each opt-in and
independently disableable. There is no “neural mode passed” blanket approval.

### Phase 6 — explainable intent planner

1. operation vocabulary and intent schema;
2. tool-only plan generation;
3. explanation and confidence;
4. user A/B preference profile;
5. adversarial intent/safety testing;
6. no direct pixel generation;
7. human confirmation for destructive or creative operations.

**Completion boundary:** planner proposals improve workflow without bypassing policy,
and the deterministic renderer remains the sole pixel authority in the Truth Core.

### Phase 7 — Generative Studio research

1. separate UI/artifact/manifest contract;
2. licensed model and data review;
3. source-versus-synthesis disclosure UX;
4. critical identity/likeness review;
5. generated-detail provenance;
6. creative feature-specific corpora;
7. no default/fallback integration.

**Completion boundary:** users and reviewers can identify the creative boundary, all
artifacts are labeled, and no generative result can be mistaken for the default
photographic render.

---

## 12. Candidate technology decision matrix

| Candidate | Best use in Retouch | Track | Main blocker before adoption |
|---|---|---|---|
| Current BiSeNet logits | Immediate posterior/calibration foundation | Truth Core | Hard per-face postprocess and no calibrated uncertainty |
| SegFace | Parser challenger, especially long-tail/accessory regions | Research → possible Truth Core | Retouch corpus proof, ONNX/runtime work, dataset/license provenance |
| FaRL | Face-specific representation/parsing challenger | Research | Runtime, model/data license, marginal value over narrower parser |
| DECA | Optional normals/visibility/expression detail evidence | Research analyzer | Non-commercial scientific license, runtime, single-image ambiguity |
| MICA | Geometry evaluation/reference | Research analyzer | Deployment need, licensing, added value for subtle 2D retouch |
| TRUST/HUST | Albedo/lighting research and fairness stress tests | Research analyzer | Generative priors, licensing, source-pixel boundary, controlled validation |
| Multi-frame appearance capture | Subject calibration from short capture | Research → shoot mode | Capture requirements, runtime, privacy, deployable implementation |
| Current NAFNet | Denoise benchmark baseline | Bounded Learned Assist | Face-specific texture/mark/residual certification |
| Restormer | Denoise/deblur challenger | Research → bounded assist | Model/task/license/runtime and Retouch corpus proof |
| InstantRetouch bilateral renderer | Learned low-frequency/instruction operator research | Research → possible bounded assist | Training provenance, exact support, color/material/identity validation |
| Neutral-gray modulation model | Lightweight source-detail-preserving tone proposal | Research → possible bounded assist | Independent replication and broader evidence |
| BPFRe/RetouchFormer | Blemish proposal/repair benchmark | Research | Synthesis, mark policy, identity, real-corpus generalization |
| HGFR spectral restoration | Frequency/multi-resolution design reference | Research | Mostly synthetic paired data and production provenance |
| CodeFormer/NTIRE diffusion arms | Severe degradation comparison | Generative Studio research | Invented detail, quality/fidelity tradeoff, labeling and licensing |
| BeautyGRPO | Creative face retouch and review-rubric research | Generative Studio | Generative identity risk, preference bias, base-model license |
| ArcFace/AdaFace | Identity drift alarms | QA only | Retouch-specific calibration; never sufficient alone |

No candidate is approved for implementation or shipping by this matrix.

---

## 13. Proposed module boundaries

Names are provisional and document ownership, not authorized code.

| Proposed boundary | Responsibility | Must not do |
|---|---|---|
| `face_observation` | Detector/parser/landmark/visibility evidence | Decide edits |
| `region_posteriors` | Calibrated semantic probabilities and boundaries | Hide fallback arm |
| `face_diagnosis` | Material, degradation, mark, lighting observations | Mutate pixels |
| `operation_plan` | Resolve user intent, policy, budgets, conflicts | Run model or edit |
| `operation_result` | Proposed pixels, support, residual, evidence | Expand support silently |
| `material_evidence` | Coordinated intrinsic/chromophore/specular/makeup/texture analysis | Claim medical truth |
| `subject_calibration` | Job-local multi-frame stability and consistency | Create persistent people database |
| `bounded_restore` | Guarded learned operation adapters | Use generative fallback |
| `face_fidelity_qa` | Isolation, color, texture, geometry, identity alarms | Certify alone |
| `generative_studio` | Explicit creative/synthetic rendering | Enter default pipeline |

The current modules should be adapted incrementally. A big-bang rewrite is not
recommended.

---

## 14. Release and rollback design

Every phase ships behind independent controls:

- evidence collection can ship without visible changes;
- each material operation has a current-classical fallback;
- parser challengers remain selectable in the benchmark until certified;
- subject calibration is optional;
- each learned assist has its own enable flag and model record;
- Generative Studio is a different mode, not a flag buried in a recipe;
- previous best recipe/render path remains reproducible during A/B;
- rollback means disabling one bounded change, not reverting the entire face engine.

Required release records:

- source revision and dirty/clean state;
- corpus and annotation versions;
- model/data/license provenance;
- calibration report;
- exact and calibrated thresholds;
- test and mutation results;
- per-condition metric vector;
- blinded human review result;
- known limitations and unmeasured conditions;
- rollback instructions;
- output/manifest schema versions.

---

## 15. What “highest face retouch we ever built” is allowed to mean

The phrase may be used internally only after a future candidate satisfies all of the
following:

1. It is compared, pairwise and blinded, against the current best certified Retouch
   path—not an old or deliberately weak baseline.
2. It wins or meets the pre-registered preference target from a powered study.
3. It is non-inferior on every hard fidelity/safety pillar.
4. It introduces no critical identity, likeness, geometry, mask, mouth, eye, color,
   or delivery defect.
5. Its operation isolation and mutation tests pass.
6. Its parser/material/learned thresholds were calibrated without using the sealed
   test set.
7. Important observable conditions show no hidden regression; underpowered or missing
   conditions are disclosed.
8. Two-reviewer plus adjudication requirements pass.
9. Model, data, and asset licenses are approved for the intended distribution.
10. The result is reproducible from recorded source, recipe, runtime, model hashes,
    provider, and export policy.

Externally, claims must remain narrower:

- “our best measured Retouch face engine on corpus version X” is supportable;
- “globally state of the art” requires an independent, current, comparable benchmark;
- “identity preserving” must name the evidence and limitations;
- “source-faithful” is reserved for the Truth Core and bounded assists that pass the
  source-fidelity contract;
- “generative” is disclosed wherever source pixels/details may have been synthesized.

---

## 16. Completion boundaries for this research task

### Complete in this document

- current-tree face-quality re-baseline;
- correction of stale July assumptions;
- current 2025–2026 face-retouch research review;
- parsing, inverse-rendering, restoration, identity, and evaluation mapping;
- three-track product boundary;
- detailed architecture and module ownership;
- corpus and annotation specification;
- exact, calibrated, human, and mutation gates;
- prioritized implementation sequence;
- model/candidate decision matrix;
- evidence-tiered facial appearance and microtexture contract;
- long-tail anatomy, accessory, occlusion, and unknown-pixel ownership;
- calibrated abstention and learned-operation risk classes;
- professional product capability reconnaissance;
- shoot-level subject intelligence and privacy boundary;
- current ICC/Exif/HDR/C2PA delivery research;
- explicit definition of the future quality claim.

### Not implemented or proven

- no engine or test code;
- no parser posterior change;
- no parser challenger download or benchmark;
- no 3D/albedo model integration;
- no subject calibration;
- no mark ledger;
- no unified material inference;
- no optical eye/lip/teeth implementation;
- no new learned restorer;
- no preference planner;
- no Generative Studio;
- no corpus acquisition or annotation;
- no newly calibrated thresholds;
- no model/data legal approval;
- no rerendered user images;
- no new automatic or human acceptance evidence;
- no claim that the current or future engine is globally highest quality.

### Authorization boundary

This document authorizes nothing by itself. Implementation should begin only from a
separately approved, narrow phase and file scope. The recommended first authorization
is Phase 0 from the existing AMG full-v2 engineering plan, followed by Phase 1 evidence
foundation. Do not begin with a generative model integration.

---

## 17. Deep-research continuation: what the second pass changes

The first pass established the product tracks and the broad engineering sequence. A
second primary-source and product-capability pass was performed to test whether that
plan was ambitious enough. It focused on skin reflectance and microgeometry,
long-tail face semantics, calibrated uncertainty, high-resolution restoration,
cross-image consistency, professional workflow expectations, color/HDR standards,
and provenance.

The result does not overturn the three-track decision. It makes the Photographic
Truth Core more rigorous and gives Bounded Learned Assist a clearer admission test.

### 17.1 Findings that materially sharpen the design

| Finding | Evidence | Engineering consequence |
|---|---|---|
| A single uncontrolled RGB image cannot uniquely separate illumination, diffuse color, specular response, roughness, subsurface scattering, makeup, and geometry | Modern appearance-capture work needs sparse views, a short head turn, scene lighting evidence, diffusion priors, or controlled capture | Retouch must attach an evidence level to every material estimate and must not present a prior-driven estimate as measured skin truth |
| Three images or a short head-turn sequence can stabilize facial geometry/reflectance estimates substantially | SFDM reports sparse-view reconstruction from as few as three images; ICCV 2025 monocular appearance capture uses a simple head rotation | Shoot mode can build a job-local appearance baseline without replacing the rendered face |
| Long-tail semantic classes are exactly where face editing fails | SegFace reports large gains for eyeglasses, earrings, and necklaces and identifies poor lighting, multiple faces, and occlusion as difficult cases | The production ontology must include accessories and occluders, not just skin/eyes/lips/hair |
| Generic and face-specific quality scores do not reliably cover authenticity, identity, and local facial defects | F-Bench/FaceQ uses multi-dimensional human ratings; restoration benchmarks report disagreement between common metrics and human judgement | No scalar automatic score may certify a face render |
| Segmentation confidence can be calibrated into prediction sets or abstention policies | Conformal segmentation research provides post-hoc uncertainty sets with empirical coverage goals | Parser uncertainty should control apply/attenuate/preview/skip decisions rather than be logged passively |
| Cross-image editing and subject-level consistency are now active model research areas and shipping product features | Group Editing studies multiple images jointly; Evoto exposes subject-level batch synchronization | Retouch needs shoot-level subject association and consistency evidence, but generative group editing remains outside the default path |
| Major commercial tools openly document failures behind glasses, on profile views, around teeth, facial hair, fingers, jewelry, and darker skin | Adobe's published Neural Filter known-issues page names these conditions | These conditions belong in the required corpus and critical-defect taxonomy, not an optional edge-case list |
| Current provenance standards distinguish retained bytes from a valid derived-work assertion | C2PA defines signed claims, ingredients, actions, validation states, and update manifests | Copying an old APP11 block is not proof of valid provenance after pixels change |

### 17.2 Revised definition of the quality ceiling

The future engine must maximize five quantities simultaneously:

1. **correction success** — the requested distraction is genuinely reduced;
2. **evidence fidelity** — unchanged identity/material evidence remains source-derived;
3. **uncertainty honesty** — the system abstains or weakens when it cannot localize or
   explain an operation;
4. **workflow consistency** — the same subject and shoot remain coherent without
   forcing identical treatment onto unlike frames;
5. **delivery truth** — exported pixels, profiles, metadata, provenance status, and
   evidence records agree.

The first pass emphasized source pixels. The second pass adds an equally important
rule:

> Authority must be proportional to evidence. More sophisticated inference does not
> grant permission to make a stronger edit.

---

## 18. Production facial-appearance model

### 18.1 What an observed face pixel contains

An ordinary portrait pixel is a camera-rendered mixture of:

- illumination color, direction, size, distance, and indirect bounce;
- diffuse surface reflectance;
- spatially varying specular intensity and roughness;
- subsurface transport;
- mesostructure such as wrinkles, folds, and larger pores;
- microstructure such as fine pores and vellus hair;
- makeup, sunscreen, oil, sweat, powder, glitter, and other surface layers;
- facial/scalp hair and accessory occlusion;
- camera spectral sensitivities, white balance, demosaic, denoise, sharpening,
  tone curve, local tone mapping, compression, and display conversion.

This makes a single-image “skin truth” claim underdetermined. Even modern systems that
produce attractive diffuse/specular/normal maps may use learned priors to fill missing
evidence. SFDM explicitly separates geometry, diffuse reflectance, specular
reflectance, and subsurface effects from sparse views; S3-Face uses diffusion priors
to estimate SSS-compliant reflectance; and Monocular Facial Appearance Capture models
visibility, geometry, diffuse albedo, specular intensity, roughness, and unknown
environment lighting from a short video.

Retouch should use these papers as a decomposition vocabulary and research benchmark,
not as permission to call inferred layers ground truth.

### 18.2 Evidence levels and allowed behavior

Every face observation receives an evidence level independent of edit strength.

| Level | Input evidence | Allowed diagnosis | Default edit authority |
|---|---|---|---|
| `E0_RENDERED_SINGLE` | One JPEG/HEIC/TIFF with unknown rendering | Relative local contrast/color, visible texture, visible geometry, parser/landmark evidence | Conservative; no absolute pigment, roughness, or albedo claim |
| `E1_DECLARED_SINGLE` | One RAW or fully declared color-managed image with usable scene context | Better linear-light, degradation, exposure, and relative material evidence | Bounded single-frame material correction with confidence limits |
| `E2_SHOOT_MULTI` | Several registered frames of the same subject across pose/exposure/lighting | Stable marks, recurrent texture, view-dependent highlight evidence, shot outliers | Stronger diagnosis; still render from each frame's own source pixels |
| `E3_CALIBRATED_CAPTURE` | Controlled chart/gray reference, known illuminants, polarization or validated capture procedure | Calibrated color/reflectance research and threshold development | Research/reference authority; production editing still follows user intent |

Rules:

- an `E0` face may never be promoted because a learned model looks confident;
- a higher evidence level permits stronger diagnosis, not automatic stronger beauty
  treatment;
- evidence level is per face and per property: color may be `E1` while occluded lip
  semantics remain unknown;
- the manifest records the level, missing evidence, fallbacks, and any downshift;
- no ordinary portrait receives medical pigment, diagnosis, or Fitzpatrick labels.

### 18.3 Material-component ownership

| Observed component | Source-faithful operation | Forbidden shortcut in Truth Core |
|---|---|---|
| Low-frequency diffuse variation | Gradual, mask-bounded tone/chroma correction with form protection | Flattening all cheeks toward one average color |
| Specular highlight | Modify intensity/roughness proxy while retaining highlight footprint and lighting logic | Painting a generic dewy/matte lobe unrelated to source geometry |
| Shadow/form | Low-band correction constrained by geometry and scene light confidence | Treating every dark region as pigmentation or under-eye defect |
| Pores and fine wrinkles | Preserve or reweight source high-frequency residual locally | Synthesizing generic pores and calling them recovered detail |
| Blemish/temporary mark | Policy-authorized gradual correction with stable-mark exclusion | Removing all spots selected by darkness or a beauty classifier |
| Makeup/coating | Detect and protect or adjust as a surface layer | Treating foundation, blush, glitter, or face paint as bare skin |
| Facial/vellus hair | Preserve as an occluding fiber layer unless explicitly targeted | Blurring it into skin or using it as texture donor |
| Severe missing/degraded detail | Report insufficient evidence; offer separate creative/restoration path | Hallucinating detail in the default output |

### 18.4 Pore and microtexture contract

Fine-scale skin structure is region-dependent. Nose, forehead, cheeks, periocular skin,
and neck do not share one stationary pore pattern. Capture/rendering research has long
shown that facial microgeometry and its specular response materially affect realism.

The Truth Core therefore follows these rules:

1. Source high-frequency content is never replaced wholesale.
2. Texture transfer, when permitted, stays within the same subject, compatible region,
   scale, orientation, focus plane, and lighting context.
3. Repeated/copied patches are detected through local correlation and nearest-neighbor
   duplication checks.
4. “Pore synthesis” is renamed or confined to Generative Studio unless it only
   reweights existing source residual.
5. Texture metrics are stratified by facial region and face-pixel scale; one global
   Fourier score is insufficient.
6. Small faces below the measurable pore scale must not receive a positive
   pore-preservation claim.
7. Sharpening cannot be used to manufacture pore-like ringing.

Required microtexture evidence:

- high-band energy ratio and orientation distribution;
- local phase/correlation retention against source;
- repeated-patch alarm;
- edge overshoot/undershoot around pores, hairs, lips, and wrinkles;
- source/output crops at native 100%, 200%, and normal viewing size;
- reviewer labels for wax, crunch, stamped texture, pore disappearance, and false
  detail.

---

## 19. Semantic ownership and occlusion model

### 19.1 The production ontology must be editing-oriented

A generic face parser's classes are not sufficient. Retouch needs classes based on
whether pixels may be changed by a particular operation.

| Region or material | Default ownership | Required behavior |
|---|---|---|
| Exposed facial skin | Operation-specific editable candidate | Still exclude marks, hair, makeup, folds, boundaries, and uncertain pixels |
| Neck/ear/body skin | Separate material and exposure regions | Diagnose face/body mismatch; never assume face settings transfer directly |
| Lip vermilion | Lip operation only | Separate diffuse color, texture, lines, and specular moisture |
| Oral aperture/interior | Protected by default | Never inherit lip or skin tone operations |
| Teeth | Teeth operation only | Preserve tooth individuality, edge translucency, shadows, restorations, and gaps |
| Gingiva/gums | Protected | No whitening; color change is a critical defect |
| Tongue | Protected | No whitening, lip recolor, or skin correction |
| Saliva/wet highlights | Protected optical evidence | Avoid neutralizing or converting them into tooth pixels |
| Sclera | Eye operation only | Redness/brightness bounds; preserve lid shadows and wet appearance |
| Iris/pupil | Eye operation only | No recolor or synthetic detail without explicit creative intent |
| Corneal/catchlight layer | Protected or tightly bounded | Preserve source light direction, number, shape, and occlusion logic |
| Eyelids/lashes/brows | Distinct hair/skin boundaries | Never treat lashes/brows as blemishes or skin texture |
| Facial/scalp hair | Protected fiber layer | Prevent smoothing, tone equalization, and texture transfer bleed |
| Makeup/face paint/glitter | Protected coating unless explicitly targeted | Preserve character design, finish, edges, and deliberate asymmetry |
| Glasses lens/frame/glare | Occluder plus optical layer | Do not edit hidden skin; glare removal is separate and evidence-limited |
| Piercing/jewelry/earrings | Protected object | No geometric mutation, disappearance, or skin blending |
| Hand/fingers | Foreground occluder | Face operation must stop underneath; no reconstruction behind it |
| Mask/veil/lace/microphone/costume | Protected occluder | Keep structure and material; do not infer hidden face |
| Tattoo/scar/beauty mark/freckle | Mark-policy governed | Stable/persistent evidence wins over generic blemish classification |
| Unknown/ambiguous | Protected unknown | Attenuate, skip, or request manual support |

SegFace is relevant because it treats eyeglasses, hats, earrings, and necklaces as
long-tail classes and reports that class-specific modeling helps. Its taxonomy is
still incomplete for Retouch; it is a challenger and ontology reference, not a drop-in
solution.

### 19.2 Tri-state pixel ownership

Every operation resolves three masks, not one:

1. `editable_support` — pixels the operation is permitted to propose changing;
2. `protected_support` — known pixels it must restore exactly from source;
3. `unknown_support` — uncertain pixels that default to protected and are surfaced in
   evidence.

The final support is never simply “parser probability above threshold.” It is the
intersection of user intent, semantic eligibility, material eligibility, visibility,
mark policy, confidence, and operation-specific budgets, minus protected and unknown
support.

### 19.3 Occlusion rules

- Truth Core edits only visible evidence; it never completes the hidden face.
- Soft feathering may cross an uncertain boundary only if the final exact source
  restore removes all unauthorized residual.
- An occluder mask may be coarse during proposal generation, but final support must be
  conservative at hair, glasses, jewelry, hand, mouth, and costume boundaries.
- Multi-frame evidence may reveal what lies behind an occluder for diagnosis, but the
  current frame's hidden pixels remain untouched.
- Removing glasses glare, stray hair, or a hand is a distinct object-removal request
  and does not inherit authorization from skin retouch.
- If an accessory changes shape, count, color, or visibility without explicit intent,
  the result is rejected as a critical object-fidelity defect.

### 19.4 Mouth-specific semantic minimum

ICCV 2025 phone-based teeth reconstruction highlights the hard combination of dental
segmentation, similar-looking teeth, lip occlusion, and teeth/lip interpenetration.
Retouch does not need a dental avatar to whiten teeth, but it does need an ontology
that respects the same boundaries.

Minimum mouth evidence:

- upper/lower lip vermilion;
- lip skin transition;
- oral aperture;
- visible individual-tooth support or conservative aggregate teeth support;
- gum support;
- tongue support;
- braces, retainers, grills, jewelry, lipstick transfer, food, and unknown objects;
- mouth openness, visibility, blur, and specular confidence.

When tongue evidence is missing, whiten only a high-confidence tooth core or skip.
Never derive a broad teeth mask from brightness alone.

---

## 20. Calibrated uncertainty and abstention

### 20.1 Why a confidence number is not enough

Modern neural outputs are often miscalibrated, especially after changes in camera,
makeup, face scale, pose, lighting, accessories, or compression. A reported `0.95`
does not mean a 95% chance that a mouth or eye mask is safe.

Conformal segmentation research demonstrates a useful direction: use a held-out
target-domain calibration set to produce prediction sets with measurable empirical
coverage. This does not make Retouch formally safe by itself—the corpus and loss must
match our editing risks—but it is more defensible than copying a threshold from a
paper.

### 20.2 Operation decision states

Each high-risk operation returns one of five states:

| State | Meaning | Pixel behavior |
|---|---|---|
| `APPLY` | Required semantics and boundaries are reliable for this strength | Render inside certified support |
| `ATTENUATE` | Core support is reliable but boundary/material evidence is weaker | Reduce strength and shrink toward high-confidence core |
| `PREVIEW_ONLY` | Proposal may help a human but automatic release is not justified | Produce labeled preview; do not include in unattended final |
| `SKIP_SAFE` | Evidence is insufficient or protected/unknown overlap is too high | Return source pixels and a reason |
| `MANUAL_REQUIRED` | User intent is clear but support cannot be safely inferred | Require manual mask/confirmation before render |

No fallback may convert `SKIP_SAFE` into a broader landmark or global operation while
reporting success.

### 20.3 Calibration program

For every parser/material/learned decision:

1. define the harm-weighted error before looking at thresholds;
2. lock train, calibration, development, and sealed-test identities separately;
3. calibrate by operation and critical class rather than mean parser score;
4. measure reliability and coverage by face scale, pose, blur, lighting, skin
   appearance, makeup, occlusion, accessory, and runtime arm;
5. report coverage versus support size and abstention rate;
6. test covariate shifts such as another camera, export profile, convention makeup,
   phone compression, and low light;
7. automatically downshift to a safer state when drift/OOD signals rise;
8. recalibrate after model, preprocessing, class map, provider, or precision changes;
9. retain a non-learned fail-open path;
10. mutation-test that uncalibrated logits and falsely high confidence cannot pass.

### 20.4 Uncertainty is property-specific

A face can simultaneously have:

- high detector confidence;
- high skin-core confidence;
- low lip/teeth boundary confidence;
- unknown tongue evidence;
- low lighting confidence;
- medium mark stability;
- high identity association confidence.

The engine must not compress that vector into one “face quality” scalar. Operation
eligibility consumes only the relevant properties and records why it applied,
attenuated, or abstained.

---

## 21. Current professional-product benchmark

This is a capability reconnaissance based on official vendor documentation, not an
independent output-quality ranking. Vendor claims are treated as product-positioning
evidence until Retouch runs a controlled source/output bake-off.

| Product/workflow | Officially documented capability | Lesson for Retouch | Boundary |
|---|---|---|---|
| Adobe Photoshop | Non-destructive Neural Filter outputs, skin smoothing, portrait transformation, masked/new-layer options | Preview, rollback, and isolated outputs are table stakes | Adobe also documents failures around body skin, glasses, profiles, jewelry, teeth, facial hair, fingers, occlusion, and darker skin; do not copy feature claims without testing these conditions |
| Capture One | Layer-based local color, heal and clone, manual source-point control, 100% inspection | Professional users need an editable layer/operation model and the ability to override automatic source/support | It is primarily a controlled manual workflow, not proof that automation is safe |
| Retouch4me | Modular Heal, Dodge & Burn, Skin Tone, Mattifier, eyes, teeth, mask, hair, and other task-specific plugins | Separate operation modules and independent intensity controls are preferable to one opaque “beautify” model | Marketing naturalness claims require our own corpus comparison |
| Evoto | Subject selection, per-subject adjustments, batch synchronization, skin/face/eyes/teeth/makeup/hair/hands modules | Subject-level shoot workflow and region-specific controls are competitive table stakes | Automatic age/gender tagging must not become Retouch's policy or default-strength mechanism |
| MeituYunxiu | RAW conversion, large batch workflow, AI skin detail, face/body changes, presets, export | Throughput, full-job consistency, exception handling, and RAW-to-delivery integration matter as much as one-image quality | Vendor feature count and speed claims do not establish source fidelity or identity safety |

### 21.1 Competitive conclusions

Retouch should not compete by adding the largest number of beauty sliders. Its
credible differentiation is:

- source-authoritative, evidence-producing edits;
- explicit stable-mark and object preservation;
- per-operation calibrated abstention;
- shoot-level consistency without demographic beauty defaults;
- exact support and color/provenance truth;
- visible separation of photographic and generative work;
- reproducible automatic and human certification.

The competitor bake-off must compare the same RAW or source files under controlled
export settings. Required artifacts are source, vendor output, Retouch output,
registered crops, residuals, metadata/profile inspection, timing, manual intervention,
and blinded review. Screenshots and vendor before/after examples are not benchmark
evidence.

### 21.2 Feature gaps that are workflow-important but not automatically quality wins

- per-subject batch synchronization;
- manual subject correction and missing-face addition;
- manual support/source override;
- editable operation stack or sidecar;
- hair, hand, glasses, jewelry, and body-skin ownership;
- batch exception queue and rerender only failed/outlier frames;
- source/output comparison at native zoom;
- preset versioning and project-level consistency controls.

These should enter the product roadmap only with the same safety contract as the
render engine.

---

## 22. Learned-operation risk classes

“Uses AI” is not a meaningful engineering category. Learned candidates are classified
by what they are allowed to output.

| Class | Output | Example | Default track | Required control |
|---|---|---|---|---|
| `L0` | Deterministic measured transform | Curves, local chroma, exact mask composite | Truth Core | Color/support/reconstruction tests |
| `L1` | Learned observation only | Parser posterior, blemish proposal, degradation estimate | Truth Core candidate | Calibration, OOD, no direct pixel authority |
| `L2` | Bounded transform parameters | Bilateral-grid coefficients, low-frequency gain/color field | Bounded Assist candidate | Parameter bounds, support lock, source application, rollback |
| `L3` | Residual or restored pixels | Denoiser/deblur/restoration network | Bounded Assist research | Frequency/residual/mark/identity budgets and exact source blend |
| `L4` | Prior-generated or inpainted content | Codebook, diffusion restoration, generative relight/retouch | Generative Studio | Disclosure, provenance, creative review, never default fallback |

### 22.1 Model admission contract

Every candidate model record must contain:

- operation and output-risk class;
- architecture and source repository;
- exact weight hash and acquisition source;
- code, weight, training-data, and downstream-use license review;
- training-data provenance and known synthetic/real proportions where available;
- preprocessing, color/transfer assumptions, input range, precision, and resolution;
- provider/device and deterministic/nondeterministic status;
- runtime, memory, tiling, border, and failure behavior;
- calibrated corpus and sealed-test results;
- condition-specific limitations;
- fallback behavior and evidence truth;
- versioned disable/rollback control.

No model ships only because its repository license is permissive. Code, weights,
training data, base model, dependencies, and intended distribution are separate
questions.

### 22.2 High-resolution inference contract

Face retouch often exposes failures that 512- or 1024-pixel benchmarks hide. A learned
operation must prove:

- native face-scale consistency across small, medium, and large faces;
- no tile seams, context discontinuities, padding halos, or repeated detail;
- stable results when crop origin moves by a few pixels;
- consistent support when the same face is rendered alone and inside the full image;
- no silent downscale/upscale advertised as native restoration;
- deterministic repeatability where claimed;
- bounded behavior when model/provider/runtime changes;
- memory failure returns source, not a degraded partial result;
- per-band residuals are within the operation's declared authority.

### 22.3 Reference-photo restoration

Recent personalized restoration research uses additional photographs of the subject.
This can improve apparent identity, but it adds risks:

- wrong-person association;
- transplanting age, makeup, facial hair, expression, or lighting from another date;
- privacy and biometric retention;
- using a reference to invent detail not visible in the target;
- cross-subject leakage in group jobs.

In Truth Core, references may improve diagnosis and stable-mark protection. They do
not authorize transplanting facial pixels. Any reference-conditioned pixel generation
remains Generative Studio until a narrower source-fidelity contract is proven.

### 22.4 Evaluators cannot certify their own editor

F-Bench, BeautyGRPO, HP-Edit, and recent editing agents show the value of learned
preference/evaluation models. They also expose a circularity risk: optimizing an
editor against a learned evaluator can reward its blind spots.

Rules:

- a learned evaluator is an alarm or ranking feature, never sole release authority;
- editor and evaluator must not share undisclosed training leakage;
- evaluator performance is calibrated against Retouch's human labels;
- mutations deliberately target likely reward hacks: smoothness, whitening, false
  sharpness, symmetric faces, copied pores, and identity-normalized features;
- final human reviewers remain independent from model training and threshold tuning.

---

## 23. Evaluation system extensions

### 23.1 Nine-layer evidence stack

| Layer | Question | Examples |
|---|---|---|
| 1. Mechanical | Did the operation change only authorized pixels? | Exact outside-support residual, zero-strength identity, determinism |
| 2. Representation | Were pixels interpreted and exported correctly? | Transfer function, gamut, profile, precision, metadata |
| 3. Semantic | Was the intended anatomy/material selected? | Critical-class precision/recall, boundaries, protected overlap |
| 4. Correction | Did the requested issue improve? | Blemish visibility, blotch/form separation, glare/redness reduction |
| 5. Material | Does skin/eye/lip/teeth behavior remain physically plausible? | Specular footprint, texture, translucency proxies, wet/dry boundaries |
| 6. Fidelity | Did the subject and personal evidence remain intact? | Marks, contours, local correlation, identity alarms, human likeness |
| 7. Photographic | Does the full portrait still read as one photograph? | Face/body, light direction, noise, depth of field, scene harmony |
| 8. Shoot | Is the job coherent without flattening frame differences? | Subject consistency, outlier rate, group tone, per-shot intent |
| 9. Human | Is the result preferred and free of critical defects? | Blinded paired review, regional labels, adjudication |

No layer substitutes for another. A visually preferred candidate with outside-support
drift fails. A pixel-exact candidate that does not correct the requested issue also
fails.

### 23.2 Automatic metric roles

- PSNR/SSIM: regression and reconstruction diagnostics, not naturalness.
- LPIPS/DISTS-like perceptual distance: change alarm, not identity or correctness.
- ArcFace/AdaFace: identity-drift alarm, not proof of likeness.
- generic/face IQA: artifact/quality ranking feature, not beauty authority.
- FID/distribution metrics: corpus-level comparison only, never individual
  source-fidelity evidence.
- learned face evaluators: multi-dimensional candidate triage after Retouch-specific
  calibration.
- exact residuals and metadata/profile checks: hard mechanical gates where equality
  is required.

F-Bench is especially useful for its separation of authenticity, identity fidelity,
and other face-quality dimensions. It is oriented toward generated/customized/restored
faces, so its labels and evaluator are references for taxonomy—not automatic Retouch
certification.

### 23.3 Fairness and observable-condition testing

Recent face-quality research reports that luminance-based quality measures can reject
darker-appearing faces disproportionately. Facial-hair studies also show that
recognition behavior changes with hair and that dataset balancing alone does not erase
the gap.

Retouch must therefore:

- test quality alarms by continuous observed luminance, chroma, SNR, contrast, and
  face scale rather than inferred race;
- include facial hair, makeup styles, age-related texture, darker and lighter image
  appearance, glasses, accessories, and lighting extremes;
- collect consented demographic labels only when a justified fairness study requires
  them and keep them separate from render policy;
- never set beauty strength from inferred gender, age, race, ethnicity, or medical
  skin type;
- report false apply, false skip, and critical-defect rates by observable condition;
- investigate a gap instead of “fixing” it by normalizing the subject toward the
  training majority.

### 23.4 Human-review protocol additions

Review happens in two viewing modes:

1. **normal delivery view** for overall photographic coherence;
2. **native 100% inspection** for texture, boundaries, copied detail, and small
   material defects.

Reviewers answer separate questions for:

- intended correction;
- personal-feature preservation;
- skin texture/material;
- eye/lip/teeth realism;
- geometry/expression;
- accessories and occlusion;
- color/lighting;
- face/body/shoot coherence;
- visible synthesis;
- overall preference.

Do not ask only “which is more beautiful?” That wording collapses correction,
identity, realism, style, and demographic preference into one biased label.

### 23.5 New mutation families

Add these mutations to the existing program:

- mark a prior-generated material map as measured evidence;
- promote `E0` input to calibrated evidence based on model confidence;
- turn unknown support into editable support;
- remove glasses/hand/jewelry protection;
- apply face settings directly to neck/body skin;
- synthesize generic pores and preserve aggregate high-frequency energy;
- pass a copied-pore output because its spectrum matches;
- report a tiled/downscaled learned result as native resolution;
- swap subject associations within a group batch;
- apply identical strength to all frames despite differing evidence;
- use inferred age/gender to select default beauty strength;
- let an editor pass solely because its own evaluator prefers it;
- preserve an old C2PA APP11 payload and report the modified asset as validly signed;
- export HDR-tagged pixels from an SDR processing path.

---

## 24. Shoot-level subject intelligence

### 24.1 Goal

The goal is not a persistent face-recognition database. It is a job-local evidence
graph that helps Retouch distinguish stable subject properties from frame-specific
lighting, expression, focus, occlusion, and temporary distraction.

### 24.2 Subject graph

For each shoot:

- detect face instances;
- propose local subject clusters using embeddings plus time, clothing, scene, and
  manual confirmation;
- expose merge/split/correct controls;
- assign an anonymous job-local subject ID;
- record association confidence and ambiguity;
- compute stable marks, recurrent texture, makeup, facial-hair, color, and degradation
  evidence only after enough consistent frames;
- destroy or export the graph according to an explicit project retention choice.

Embeddings alone do not authorize a merge. A wrong merge can propagate one person's
retouch policy to another, so uncertain clusters remain separate.

### 24.3 Consistency is not identical treatment

Shoot consistency means:

- stable personal marks receive the same policy;
- repeated lighting/color errors receive coherent correction;
- skin texture and finish stay recognizably the same subject;
- the same requested style has comparable perceptual strength;
- outlier frames are identified and reviewed.

It does **not** mean copying the same numeric parameters or pixels to every frame.
Different pose, face scale, exposure, expression, occlusion, and focus require
frame-specific support and backoff.

### 24.4 Multi-image evidence use

Allowed in Truth Core:

- stabilize mark classification;
- estimate whether a highlight is view-dependent;
- distinguish transient compression/noise from persistent texture;
- detect per-frame exposure or white-balance outliers;
- improve pose/visibility confidence;
- choose a safer source patch from the same frame or verify a same-subject donor when
  explicit texture transfer is authorized.

Not allowed in Truth Core:

- replace an occluded eye or mouth with another frame;
- transplant a smile, pore field, makeup edge, or expression line;
- age-normalize the subject across dates;
- construct a canonical face and render it over every photograph.

### 24.5 Cross-image learned editing boundary

CVPR 2026 Group Editing demonstrates that joint multi-image editing and cross-view
consistency can be learned. Its output is generative and is best used as:

- a Generative Studio candidate;
- a benchmark for consistency metrics;
- evidence that group-level datasets and alignment matter.

It is not a default solution for source-faithful event photography.

---

## 25. Color, HDR, metadata, and provenance ceiling

### 25.1 SDR color contract

For every input and output:

- identify embedded profile, declared color space, transfer function, channel order,
  bit depth, alpha semantics, and orientation;
- apply the source profile before operations defined in the working space;
- perform linear-light operations only on genuinely linear values;
- retain enough precision through intermediate material and compositing operations;
- convert to the explicit delivery profile once;
- embed the delivery profile and reset orientation after physical rotation;
- verify decoded output pixels under the embedded profile;
- record gamut clipping/compression, rendering intent, export subsampling, quality,
  and metadata policy.

The ICC maintains ICC.1 v4 specifications and profile resources. The current CIPA
standards page lists Exif 3.1 and the 2026 Exif-to-XMP mapping revision. The project
must version its conformance target; “preserve EXIF” without naming orientation,
privacy, and rewritten fields is incomplete.

### 25.2 HDR is a separate product mode

An HDR face workflow is not “save the SDR result in Display P3.” It requires:

- a declared BT.2100 PQ or HLG signal path or another explicit still-image HDR
  standard;
- high-precision linear/scene or display-referred processing decisions;
- reference-white and peak-luminance policy;
- controlled tone mapping and SDR rendition;
- highlight/specular behavior designed for HDR;
- metadata/container support;
- HDR-capable viewing and review;
- cross-display and SDR fallback QA.

ITU guidance explicitly cautions against treating one skin-tone luminance as a
universal reference because skin reflectance and environments vary. Retouch should use
neutral/chart and display targets, not normalize people to an assumed skin brightness.

Until this path exists, the quality-ceiling release target remains color-managed SDR.

### 25.3 C2PA and derived-work truth

C2PA 2.4 defines a signed provenance system with ingredients, actions, claims,
content bindings, validation, and update manifests. It does not make copied metadata
cryptographically true.

Required states:

- `absent` — no C2PA material found;
- `source_present_unvalidated` — bytes found but no successful validation performed;
- `source_valid` — source manifest validated before editing;
- `legacy_bytes_preserved_unverified` — old payload copied for compatibility, not
  claimed valid for modified pixels;
- `derived_unsigned` — modified output with recorded local provenance but no C2PA
  signature;
- `derived_signed_valid` — a new derived-work/update manifest with ingredient/action
  chain validates against the output;
- `invalid` — validation attempted and failed.

The UI and manifest must never display `legacy_bytes_preserved_unverified` as a valid
Content Credential. A future signing feature requires key management, trust-list,
timestamp/revocation, privacy, action, ingredient, and validator interoperability
design—not merely APP11 copying.

---

## 26. Consolidated architecture deltas

The second pass adds or sharpens the following work packages.

| Existing phase | Second-pass insertion |
|---|---|
| Phase 0 | Add the minimum mouth/accessory protection ontology and C2PA-state truth to the existing exact-support/export work |
| Phase 1 | Build the full editing ontology, property evidence vector, calibration splits, and abstention controller |
| Phase 2 | Implement evidence levels and material/microtexture observations before enabling stronger correction |
| Phase 3 | Certify eye/lip/teeth optical regions plus hair, makeup, glasses, jewelry, hand, and unknown exclusions |
| Phase 4 | Add job-local subject graph, manual correction, retention controls, and per-frame consistency planning |
| Phase 5 | Enforce learned-operation risk classes, high-resolution laboratory, and independent evaluator controls |
| Phase 6 | Let the planner consume evidence/abstention state; never let it raise authority or self-certify |
| Phase 7 | Apply generative disclosure, provenance, reference-photo privacy, and group-editing boundaries |
| Cross-cutting | Run the professional competitor bake-off and current ICC/Exif/C2PA conformance checks before the final quality claim |

### P0.5 — editing ontology and protection graph

Deliverables:

- versioned region/material/occluder taxonomy;
- editable/protected/unknown masks;
- mouth and eye optical subregions;
- long-tail accessories and hand/hair/costume exclusions;
- cross-operation ownership conflicts;
- fail-open fallback rules;
- class-map and migration tests.

Completion:

- every face operation declares what it owns and what it must preserve;
- unknown pixels are protected by construction;
- mutations that remove protection fail direct tests.

### P0.6 — calibrated abstention controller

Deliverables:

- property-specific evidence vector;
- operation decision states;
- calibration and sealed-test splits;
- reliability/coverage reports;
- drift/OOD and fallback behavior;
- mutation-sensitive evidence truth.

Completion:

- at least mouth, eye, and learned restoration paths consume calibrated decisions;
- failure arms are correctly reported;
- critical-risk coverage and abstention targets pass on the locked corpus.

### P1.5 — evidence-tiered material engine

Deliverables:

- `E0`–`E3` evidence levels;
- diffuse/specular/form/texture/makeup/hair observations;
- component reconstruction and uncertainty;
- source-authoritative microtexture protocol;
- multi-frame material stabilization;
- claims vocabulary that distinguishes observed, inferred, and generated.

Completion:

- no output or manifest overstates input evidence;
- each material operation passes component, support, texture, mark, color, and human
  gates;
- prior-generated analyzers cannot silently provide default pixels.

### P2.3 — model governance and high-resolution laboratory

Deliverables:

- learned-operation risk classes;
- model/data/license records;
- crop/tiling/scale/determinism tests;
- editor/evaluator separation;
- resource-failure and provider-change tests;
- per-model disable and rollback.

Completion:

- only independently certified task-specific models ship;
- model absence/failure returns truthful source-safe behavior;
- no `L4` path enters default or fallback rendering.

### P2.4 — professional workflow and competitor bake-off

Deliverables:

- controlled Adobe/Capture One/Retouch4me/Evoto/Meitu comparison protocol;
- per-subject correction and batch-exception workflow specification;
- editable operation stack/sidecar design;
- timing and manual-intervention evidence;
- blinded source/output review.

Completion:

- Retouch's target advantage is measured against current products on the same corpus;
- missing table-stakes workflow is separated from unproven quality claims;
- vendor marketing examples are not accepted as comparative evidence.

### P2.5 — delivery-standard and provenance certification

Deliverables:

- named ICC/Exif conformance targets;
- profile/orientation/metadata decode verification;
- explicit SDR release contract;
- HDR research gate;
- C2PA validation-state model;
- derived-work signing research separate from byte passthrough.

Completion:

- exported pixels and manifest fields agree;
- validation mutations fail;
- no unsigned or invalid provenance is shown as signed/valid.

---

## 27. Threat model and final research boundaries

### 27.1 Critical threats

| Threat | Example | Required defense |
|---|---|---|
| Wrong subject | Batch sync applies one person's settings to another | Job-local IDs, confidence, manual merge/split, mutation test |
| Wrong material | Tongue, gum, makeup, or glasses treated as skin/teeth | Editing ontology, protected/unknown masks, critical-class corpus |
| Evidence laundering | Learned prior presented as measured albedo or recovered pore detail | Evidence levels and observed/inferred/generated provenance |
| Support leakage | Local Lab/model roundtrip changes the whole image | Exact source restoration and decoded-output residual gate |
| Beauty normalization | Age, gender, tone, facial hair, or facial shape pushed toward learned average | No demographic defaults; explicit intent; fairness and likeness review |
| Metric gaming | Smoothing or false detail improves IQA/evaluator score | Vector metrics, targeted mutations, independent human review |
| Runtime drift | Different provider/model/hash changes output silently | Model record, cache key, reproducibility, provider-change gate |
| Partial failure | Tile/OOM/model error leaves mixed or degraded output | Atomic operation result, source-safe failure, truthful manifest |
| Provenance misstatement | Copied C2PA payload shown as valid after edits | Validator-backed state model and derived-work signing boundary |
| Research leakage | Sealed-test or user images tune thresholds/models | Identity-disjoint splits, immutable test commitments, consent/retention rules |

### 27.2 What “full research” is complete now

This report now covers:

- current implementation re-baseline;
- current 2025–2026 face retouch, restoration, parsing, appearance-capture, quality,
  and group-editing research;
- professional product capability reconnaissance;
- facial appearance evidence levels;
- pore/microtexture and material ownership;
- long-tail anatomy, accessory, and occlusion rules;
- calibrated uncertainty and abstention;
- learned-operation risk classes and governance;
- extended metric, human-review, fairness, and mutation systems;
- shoot-level subject intelligence and privacy boundary;
- SDR/HDR, ICC/Exif, and C2PA delivery contract;
- additional engineering packages, threats, and completion gates.

### 27.3 What remains deliberately empirical

Research cannot responsibly preselect:

- final parser, material, restoration, or evaluator model;
- numeric confidence, conformal-risk, residual, texture, identity, or preference
  thresholds;
- morphology/feather sizes across face scales;
- minimum frame count and capture diversity for subject calibration;
- acceptable abstention rate;
- reviewer sample size and superiority/non-inferiority margins;
- commercial license approval;
- HDR delivery target;
- whether Retouch beats a named competitor.

Those answers require the locked corpus, pilot distributions, legal review where
applicable, controlled bake-offs, and human evidence. Inventing numbers in a research
document would reduce—not increase—the quality of the handoff.

### 27.4 Refresh boundary

This is a current research baseline dated 2026-09-01, not a permanent statement about
the field. Refresh it when any of these occur:

- a candidate implementation phase is authorized;
- a major parser/retouch/restoration benchmark or model changes the risk trade-off;
- the target OS/runtime/provider changes;
- ICC, Exif, C2PA, HDR, or export requirements change;
- a competitor bake-off is completed;
- corpus evidence invalidates an assumption.

The next useful work is no longer another unconstrained literature pass. It is the
Phase 0 mechanical implementation followed by Phase 1 corpus/evidence construction,
both under separate authorization.

---

## 28. Primary sources

### Face retouch and portrait consistency

- [PPR10K: portrait retouching and group-level consistency, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html)
- [FFHQR / AutoRetouch dataset and project](https://github.com/skylab-tech/ffhqr-dataset)
- [Blemish-Aware and Progressive Face Retouching, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Blemish-Aware_and_Progressive_Face_Retouching_With_Limited_Paired_Data_CVPR_2023_paper.html)
- [RetouchFormer, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28404)
- [Physics-based controllable gradual blemish retouching, ICME 2024](https://arxiv.org/abs/2406.13227)
- [Face Retouching with Diffusion Data Generation and Spectral Restoration, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Face_Retouching_with_Diffusion_Data_Generation_and_Spectral_Restorement_ICCV_2025_paper.html)
- [InstantRetouch bilateral-space project, CVPR 2026](https://openimaginglab.github.io/InstantRetouch/)
- [InstantRetouch released code](https://github.com/OpenImagingLab/InstantRetouch)
- [BeautyGRPO project, CVPR 2026](https://beautygrpo.github.io/)
- [Real-time high-resolution neutral-layer face retouching, 2026](https://doi.org/10.31881/TLR.2026.1826)

### Parsing and face evidence

- [LaPa and boundary-attention face parsing, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6832)
- [FaRL, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_General_Facial_Representation_Learning_in_a_Visual-Linguistic_Manner_CVPR_2022_paper.html)
- [SegFace official repository, AAAI 2025](https://github.com/Kartik-3004/SegFace)
- [CelebAMask-HQ data/license record](https://github.com/switchablenorms/CelebAMask-HQ)

### Geometry and appearance

- [DECA project, SIGGRAPH 2021](https://deca.is.tue.mpg.de/)
- [MICA: Towards Metrical Reconstruction of Human Faces, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730249.pdf)
- [A Morphable Face Albedo Model, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Smith_A_Morphable_Face_Albedo_Model_CVPR_2020_paper.html)
- [TRUST: scene-disambiguated skin-tone/albedo estimation, ECCV 2022](https://trust.is.tue.mpg.de/)
- [HUST: high-fidelity skin-tone/albedo estimation, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ran_HUST_High-Fidelity_Unbiased_Skin_Tone_Estimation_via_Texture_Quantization_ICCV_2025_paper.html)
- [Monocular Facial Appearance Capture in the Wild, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Monocular_Facial_Appearance_Capture_in_the_Wild_ICCV_2025_paper.html)
- [SFDM: sparse-view face geometry and reflectance decomposition, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jin_SFDM_Robust_Decomposition_of_Geometry_and_Reflectance_for_Realistic_Face_CVPR_2025_paper.html)
- [S3-Face: SSS-compliant reflectance with diffusion priors, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_S3-Face_SSS-Compliant_Facial_Reflectance_Estimation_via_Diffusion_Priors_CVPR_2025_paper.html)
- [High-Quality Facial Geometry and Appearance Capture at Home, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_High-Quality_Facial_Geometry_and_Appearance_Capture_at_Home_CVPR_2024_paper.pdf)
- [Improved Lighting Models for Facial Appearance Capture, Eurographics 2022](https://diglib.eg.org/items/d1512f3d-4924-46e6-86dd-71cf6d75d2b0)
- [Practical Measurement and Reconstruction of Spectral Skin Reflectance, CGF 2020](https://diglib.eg.org/items/5ecb6b5e-278b-4de9-9a4f-511e06e59c29)
- [Measurement-Based Synthesis of Facial Microgeometry, CGF 2013](https://diglib.eg.org/items/b76a437e-0aa3-44aa-b10d-24c49a12d5ec)

### Semantics, uncertainty, and shoot consistency

- [SegFace paper: long-tail face segmentation, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/32661/34816)
- [Teeth Reconstruction and Performance Capture Using a Phone Camera, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Teeth_Reconstruction_and_Performance_Capture_Using_a_Phone_Camera_ICCV_2025_paper.html)
- [Conformal Semantic Image Segmentation, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/html/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.html)
- [ConformalSAM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_ConformalSAM_Unlocking_the_Potential_of_Foundational_Segmentation_Models_in_Semi-Supervised_ICCV_2025_paper.html)
- [How to Trust Your Diffusion Model: conformal risk control, ICML 2023](https://proceedings.mlr.press/v202/teneggi23a.html)
- [Group Editing: Edit Multiple Images in One Go, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Group_Editing_Edit_Multiple_Images_in_One_Go_CVPR_2026_paper.html)

### Restoration and evaluation

- [NAFNet, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)
- [Restormer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)
- [CodeFormer, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf)
- [NTIRE 2025 Real-World Face Restoration challenge](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Chen_NTIRE_2025_Challenge_on_Real-World_Face_Restoration_Methods_and_Results_CVPRW_2025_paper.pdf)
- [ArcFace, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)
- [AdaFace, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)
- [LPIPS, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf)
- [The perception-distortion tradeoff, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)
- [F-Bench and FaceQ human face-quality ratings, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_F-Bench_Rethinking_Human_Preference_Evaluation_Metrics_for_Benchmarking_Face_Generation_ICCV_2025_paper.html)
- [DSL-FIQA and CGFIQA-40K, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_DSL-FIQA_Assessing_Facial_Image_Quality_via_Dual-Set_Degradation_Learning_and_CVPR_2024_paper.html)
- [Real-world Video Face Restoration benchmark, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_Towards_a_Real-world_Video_Face_Restoration_A_New_Benchmark_CVPRW_2024_paper.pdf)
- [Facial-hair accuracy bias study, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/BIOMET/html/Ozturk_Can_the_Accuracy_Bias_by_Facial_Hairstyle_be_Reduced_Through_CVPRW_2024_paper.html)
- [Demographic differentials in face-image quality, ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/CV4BIOM/html/Dorsch_Demographic_Differentials_in_Face_Image_Quality_Evaluation_and_Comparison_on_ICCVW_2025_paper.html)

### Skin-tone measurement limits

- [Prospective comparison of skin-tone scales and colorimetry, 2025](https://www.nature.com/articles/s41746-025-02245-2)
- [Limitations of ITA under uncontrolled imaging, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13459198/)

### Standards and official product references

- [ICC current specifications](https://www.color.org/specifications/)
- [CIPA standards: Exif 3.1 and Exif metadata for XMP, 2026](https://www.cipa.jp/e/std/std-sec.html)
- [ITU-R BT.2408 HDR production guidance](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-6-2023-PDF-E.pdf)
- [C2PA current specifications](https://spec.c2pa.org/specifications/)
- [Adobe Photoshop Neural Filters known issues](https://helpx.adobe.com/photoshop/using/neural-filters-feedback.html)
- [Capture One Heal layers](https://support.captureone.com/hc/en-us/articles/360002625697-Repairing-Layers-with-the-Heal-tool)
- [Retouch4me official plugin catalogue](https://global.retouch4.me/retouchplugins)
- [Evoto subject editing and batch synchronization](https://support.evoto.ai/portrait-retouching-ipad/)
- [MeituYunxiu professional batch retouching](https://yunxiu.meitu.com/?from=aigc.izzi.cn)

---

## 29. Final recommendation

Build the next face engine in this order:

1. finish mechanical correctness and delivery truth;
2. retain and calibrate soft anatomy evidence;
3. make every operation exactly isolated and representation-aware;
4. unify material diagnosis while keeping source detail authoritative;
5. make stable marks and shoot context first-class evidence;
6. improve eyes, lips, and teeth as independent optical materials;
7. benchmark learned restoration only behind residual and identity guards;
8. make operation authority proportional to declared `E0`–`E3` evidence;
9. use calibrated abstention, not raw model confidence, at critical boundaries;
10. treat accessories, occluders, makeup, hair, body skin, and unknown pixels as
    first-class ownership problems;
11. use large models for explainable planning before allowing them to generate face
    pixels;
12. keep all generative face work in a visibly separate studio mode;
13. certify SDR color, current Exif behavior, and provenance status independently of
    visual quality;
14. earn the quality claim through a locked corpus, competitor bake-off, mutations,
    blinded review, and reproducible delivery evidence.

That path is more ambitious than adding a beauty model because it improves the whole
engineering definition of face quality. It is also the only path in this research that
can plausibly produce Retouch's best face output while remaining honest about identity,
source evidence, and completion.
