# PIPELINE.md — Pipeline Architecture Reference
> Load when: planning tasks that touch pipeline stages, or when blast radius is unclear.
> Related: `docs/PERCEPTUAL.md` (pipeline invariants), `docs/VISUAL_QA.md` (gate scope).

---

## 1. Pipeline Graph

```
[Input]
  ↓
[detection.py]      RetinaFace → MediaPipe → BiSeNet cascade
  ↓
[parsing.py]        BiSeNet segmentation — skin, lips, eyes, hair region masks
  ↓
[geometry.py]       FaceReshaper — jaw/cheek/chin liquid warp
  ↓
[frequency.py]      FrequencySeparator — coarse/mid/fine band decomposition
  ↓
[skin.py]           Whitening, equalization, Dodge & Burn   [depends: parsing, frequency]
  ↓
[eyes.py]           Catchlight, dark circles                [depends: parsing]
[lips.py]           Tint, gloss, finish                     [depends: parsing]
[teeth.py]          Whitening                               [depends: parsing]
[blemish.py]        AI inpainting                           [depends: parsing, skin]
  ↓
[style.py]          Reinhard style transfer
  ↓
[grading.py]        LUT, Bloom, Split-Toning, Halation, Vignette
  ↓
[Output]
```

**Order is authoritative.** Do not call stages out of sequence. Geometry must run before frequency (warp changes the pixel field). Style must run before grading (grading operates on styled output). Features (eyes/lips/teeth) must run before style (style transfer bleeds local colors into global tone).

Pipeline invariants (what each stage may/may not do) are in `docs/PERCEPTUAL.md` §2.

---

## 2. Blast Radius Table

Use this when populating `PLAN_INPUT.blast_radius`.
Include blast radius files in the review scope, not just the directly changed file.

| Stage modified   | Downstream risk                                               | Required Visual QA scope       |
|-----------------|---------------------------------------------------------------|-------------------------------|
| `detection.py`   | All downstream stages. Cascade failure risk.                  | Full — all gates               |
| `parsing.py`     | skin, eyes, lips, teeth, blemish. Mask accuracy is critical.  | Skin Tone Uniformity, No Halo  |
| `geometry.py`    | All downstream. Warp shifts all pixel references.             | No Edge Tearing, Natural Output|
| `frequency.py`   | skin.py directly. Texture band integrity is critical.         | Texture Preservation, No Halo  |
| `skin.py`        | blemish.py (inpaints over corrected skin).                    | Texture Preservation, Skin Tone Uniformity |
| `eyes/lips/teeth`| Self-contained. No downstream dependency.                     | Natural Output (isolated)      |
| `style.py`       | grading.py operates on styled output. Color drift risk.       | No Color Drift, Natural Output |
| `grading.py`     | Final output. Clipping risk.                                  | No Highlight Clipping, No Shadow Crushing, No Color Drift |
| `params.py`      | Everything. Full regression required.                         | Full — all gates               |
| `recipe_schema.py`| Recipe validation chain.                                     | Functional test only           |

---

## 3. Module Map

| Module | Role |
|--------|------|
| `engine.py` | Pipeline orchestrator. `ProcessingContext` dataclass. JIT warmups. ThreadPoolExecutor. |
| `detection.py` | Three-stage detection cascade: RetinaFace → MediaPipe Face Mesh → BiSeNet. |
| `parsing.py` | BiSeNet ONNX face segmentation. Produces soft probability maps for each region. |
| `geometry.py` | FaceReshaper: landmark-driven liquid warp for jaw, cheek, chin shaping. |
| `frequency.py` | FrequencySeparator: decomposes image into coarse/mid/fine bands for texture-preserving skin work. |
| `skin.py` | Skin tone equalization (LAB), whitening, Dodge & Burn. Operates on frequency-separated bands. |
| `eyes.py` | Catchlight injection, dark circle reduction. Bounded to eye region mask. |
| `lips.py` | Lip tint, gloss simulation, finish (matte/satin/gloss). Bounded to lip mask. |
| `teeth.py` | Teeth whitening. Bounded to teeth mask. |
| `blemish.py` | AI inpainting for blemish removal. Operates after skin correction. |
| `style.py` | Reinhard color style transfer. Global operation, runs after local features. |
| `grading.py` | 14-stage grading: LUT, Bloom, Split-Toning, Halation, Vignette. Final look stage. |
| `params.py` | Central parameter registry. All parameters defined here. Auto-populates schema, CLI, GUI. |
| `recipe_schema.py` | JSON Schema auto-generation from params.py. Recipe validation. |
| `io.py` | EXIF copying, rawpy RAW decoding, format I/O. |

---

## 4. Model Assets

Place in `models/` before running the pipeline:

| Asset | Purpose | Required |
|-------|---------|---------|
| `face_landmarker.task` | MediaPipe landmark detector | Yes |
| `selfie_segmenter.tflite` | Hair/person mask segmenter | Yes (hair separation) |
| `resnet18.onnx` | BiSeNet face parsing ONNX model | Yes (lip, eye, skin masks) |

Missing model files are a **BLOCKER**. Do not enter the Review Loop for pipeline tasks without confirming model assets are present.

---

## 5. Recipe Contract

Every recipe is a partial override of the default parameter set.

```
RECIPE CONTRACT {
  format:           JSON matching recipe_schema.py
  partial_ok:       true — unspecified parameters inherit from defaults
  conflicts:        invalid — two keys setting the same effective value differently = schema error
  validation:       mandatory before pipeline entry (validate against recipe_schema.py)
  forbidden:        hardcoded parameter values outside of presets/ directory
}
```

Recipe-affecting changes require:
1. Schema validation: `python3 -m pytest tests/test_recipe_schema.py -v`
2. At least one named preset tested end-to-end (e.g., `film`, `cyberpunk`, `golden_hour`, `bw_noir`)