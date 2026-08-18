# Retouch Engine — Pipeline Architecture

> Reference document for the P3 stage-registry refactor. Describes the
> **current** pipeline as implemented in `retouch/engine.py` (and
> collaborators). All file:line references are to the HEAD of `main`
> unless noted. Update this doc when the pipeline shape changes.

## 1. Pipeline overview

`RetouchEngine.process()` (`retouch/engine.py:656`) is the single public
entry point. It builds a `ProcessingContext` from a recipe + caller
overrides, runs an F4 heal hook (optional), then delegates to
`_process_with_proxy()` (`retouch/engine.py:1061`), which picks a
resolution strategy (§2) and dispatches to the core pipeline.

The core pipeline is split into two phases by F8.1/F8.2:

| Phase | Method | Stages | Resolution |
|-------|--------|--------|------------|
| Detection + faces | `_run_detection_and_faces` (`engine.py:1428`) or `_process_native_faces` (`engine.py:1205`) | 0, 1, 2, 2.5 | proxy (F8.1) or native (F8.2) |
| Global | `_run_global_phases` (`engine.py:1520`) | 3, 3.5, 4, 5, 6, QA | native (always) |

`_run_core_pipeline` (`engine.py:1641`) is a thin legacy wrapper that
calls both phases in sequence at one resolution (used when no proxy
scaling is needed).

### Stage sequence

| # | Stage | Method | What it does |
|---|-------|--------|--------------|
| 0 | Detection | `_detector.detect` + `_detector.segment_person` (`detection.py:157`, `detection.py:267`) | RetinaFace bboxes + MediaPipe FaceLandmarker 478 landmarks + MediaPipe Selfie Segmentation person mask. `FaceContext` cache (`detection.py:48`) can short-circuit this. |
| 1 | Reshape | `_stage_reshape` (`engine.py:1820`) | Global face slimming/reshaping via `FaceReshaper` (`geometry.py`). Skipped when `ctx.slimming <= 0`. Runs once on the full image, before per-face crops. |
| 2 | Per-face | `_stage_per_face` (`engine.py:1836`) → `_process_face_core` (`perf_optimizations.py:228`) | For each detected face: crop an expanded ROI, parse regions (BiSeNet + landmarks), then run the per-face rendering chain — frequency separation, skin ops, blemish, undereye, eyes, teeth, lips, blush, hair, dodge/burn, shadow lift, wrinkle soften, texture transplant, local clarity. Parallelised across faces (§6). |
| 2.5 | Composite | `_composite_faces` (`engine.py:2073`) | Blend each per-face canvas back into the full image via the accumulated edit mask (`max(skin_hair, lips, sharpen)`). Accumulates `acc_skin`, `acc_skin_hair`, `acc_lips`, `acc_sharpen` masks to full-image size. |
| 3 | Subject separation | `_stage_subject_separation` (`engine.py:2113`) | LAB-L exposure shift: brighten subject (person mask), darken background. Float-native via `bgr_f32_to_lab_f32`/`lab_f32_to_bgr_f32`. |
| 3.5 | Body skin | `_stage_body_skin` (`engine.py:2173`) | LCH skin-color detection on the full body (excluding face ROI via convex hull + hair + lips), then body_smooth / body_equalize / body_whiten / body_match_face / body_relight / body_dodge_burn / body_shadow_lift / blemish. Runs a second full-image BiSeNet hair pass (`parser.parse_hair_full_image`) to exclude wigs. |
| 4 | Global tonal | `_stage_global` (`engine.py:2508`) | Contrast, brightness (gamma), highlights/shadows/whites/blacks (parametric curve), clarity, vibrance, saturation. Float-native variants `_F_adjust_*` (`engine.py:2871+`). |
| 5 | Color grading | `_stage_grade` (`engine.py:2567`) | Subject-aware color transfer, `color_ref` transfer, white balance (LCH), master HSL (LCH), `color_grade` preset or `color_grade_stack`, split toning (3-way, skin-masked), highlight rolloff, skin diffusion, bloom, glow, fade toe, highlight drift, airy haze, clarity split, post-effects (halation/grain/CA/LUT), white costume lift, vignette, film grain, negative split tone, B&W channel mixer. |
| 6 | Finish | `_stage_finish` (`engine.py:2776`) | Selective unsharp mask through `acc_sharpen` (radius scales with avg face `ied`), then `add_impact_finish`. |
| QA | Self-QA | `qa_detectors.run_all` inside `_run_global_phases` (`engine.py:1601`) | Banding / clipping / plastic-skin / halo / seam detectors run on the final uint8 output. Populates `ProcessingResult.qa` with `QAWarning` objects. |

### Data flow

```
img_bgr (uint8)
  │
  ├── fast=True? downscale to 800px (PERF-1, engine.py:786)
  │
  ├── resolve_recipe + build_context → ProcessingContext (engine.py:379)
  │
  ├── F4 heal hook (engine.py:947)           [optional, pre-pipeline]
  │
  ├── _process_with_proxy (engine.py:1061)
  │     ├── _process_native_faces (F8.2)     [stages 0-2.5 at native, 0 at proxy]
  │     │   OR
  │     ├── _run_detection_and_faces (F8.1)  [stages 0-2.5 at proxy]
  │     │     + _upscale_core_result + _composite_upscaled_faces_onto_native
  │     │     + F8.0 native-detail reinjection
  │     │
  │     └── _run_global_phases (engine.py:1520)  [stages 3-6 + QA at native]
  │
  ├── style_profile LAB skin-tone shift (engine.py:1003)  [optional]
  │
  └── ProcessingResult (engine.py:1044)
```

The unit of data passed between phases is `_CoreResult` (`engine.py:583`):
the image plus four accumulated float32 masks (`acc_skin`,
`acc_skin_hair`, `acc_lips`, `acc_sharpen`), the `faces` list, the
`person_mask`, built `face_contexts`, and `qa` warnings.

## 2. Resolution strategy (F8.1 / F8.2)

`PROXY_MAX_DIM = 2048` (`engine.py:128`). When the longest image side
exceeds this, a proxy downscale is used for the detection-heavy stage 0
so MediaPipe/RetinaFace run on a 2048px image regardless of input
resolution.

`_process_with_proxy` (`engine.py:1061`) selects the path:

### F8.2 — native face-crop path (default, `quality="full"`)

`_process_native_faces` (`engine.py:1205`).

- **Stage 0** (detection + segmentation) runs on the **proxy** image.
- Face bboxes and `ied` are rescaled to native pixel coordinates
  (`engine.py:1276`). MediaPipe landmarks are normalised `[0,1]` and are
  resolution-independent — they are *not* rescaled.
- The person mask is upscaled to native via `INTER_LINEAR`.
- Cached `face_contexts` built at proxy res are **discarded** for this
  call (`engine.py:1307`) because the cached regions are at proxy
  resolution and invalid for native face crops.
- **Stages 1–2.5** (reshape + per-face + composite) run at **native**
  resolution. Face texture and pore detail are never resampled through
  the proxy.
- **Stages 3+** run at **native** via `_run_global_phases`.

### F8.1 — legacy proxy path (`quality="draft"`)

`_run_detection_and_faces` (`engine.py:1428`) + upscale + composite.

- **Stages 0–2.5** run entirely at **proxy** resolution.
- `_upscale_core_result` (`engine.py:1364`) resizes the result image +
  masks + person mask back to native via `INTER_LINEAR`.
- `_composite_upscaled_faces_onto_native` (`engine.py:1386`) pastes only
  the edited face ROIs onto the pristine native background (background
  is never resampled).
- **F8.0 detail reinjection** (`engine.py:1132`): adds the native
  high-band (`native - GaussianBlur(native)`) back into the edit-mask
  regions, discounted by `smooth_strength` and an adaptive factor
  (`_texture_adaptation_factor` from `frequency.py`). Restricted to
  edit-mask regions only — frame-wide reinjection over-sharpens the
  background. Calibrated by `REINJECT_*` constants (`engine.py:147`).
- **Stages 3+** run at **native** via `_run_global_phases`.

### No-proxy path

When `max(h, w) ≤ PROXY_MAX_DIM`, `proxy_scale == 1.0` and both paths
collapse: F8.2 is skipped (`engine.py:1099`), F8.1 runs at native with
no upscaling, and `_run_global_phases` operates on the
`_run_detection_and_faces` output directly.

### Fast preview

Independent of F8.1/F8.2: `fast=True` downscales the input to 800px
*before* detection (`engine.py:786`, PERF-1) and upscales the result
back at the end (`engine.py:999`). This is for thumbnail/contact-sheet
use; it bypasses the proxy logic entirely because the input is already
small.

## 3. Float32 pipeline (F1 / E2)

Two distinct float conventions coexist:

### Per-face pipeline — float32 `[0, 255]` BGR

`_process_face_core` (`perf_optimizations.py:228`) converts the ROI
canvas to `float32` once at the top (`perf_optimizations.py:290`) and
keeps it float through the entire skin-operation chain. Single `uint8`
conversion at the end via `_e1_to_u8` (`perf_optimizations.py:639`).

This is the **E1** change. The convention is `[0, 255]` (matching
OpenCV BGR scale), *not* `[0, 1]`. `frequency.combine(..., float32_out=True)`
(`frequency.py:198`) returns float32 `[0, 255]` to avoid quantization in
the smoothing step.

Known uint8 boundary inside the per-face pipeline:
- `frequency.separate` expects `uint8` input
  (`perf_optimizations.py:306` — `canvas_u8_for_freq`). The canvas is
  clipped and cast to uint8 for frequency separation only, then the
  result is recombined in float.

The `_tr` / `_E1_TRACE` / `_E1_QUANT` machinery
(`perf_optimizations.py:186-225`) is a temporary bisect harness for
quantization debugging — it can force uint8 at any named op in
`_E1_CHAIN_ORDER`. Remove before shipping (per the comment at
`perf_optimizations.py:185`).

### Global pipeline — float32 `[0, 1]` BGR

`_run_global_phases` (`engine.py:1520`) converts to float32 `[0, 1]` at
the top via `precision.to_float` (`engine.py:1548`) and back to uint8
at the end via `precision.to_uint8` (`engine.py:1594`). This is the
**F1** change.

`precision.to_float` (`precision.py:31`) and `precision.to_uint8`
(`precision.py:51`) use round-to-nearest, not truncation.

Many grading/utils ops are now **dtype-aware** (F1/E2): they branch on
`img.dtype == np.float32` and use float-native paths
(`bgr_f32_to_lab_f32`, `bgr_f32_to_lch_f32`, etc. from `utils.py`).
Examples: `_stage_subject_separation`, `_grader.white_balance_lch`,
`_grader.adjust_hsl_lch`, `apply_highlight_rolloff`,
`apply_skin_diffusion`, `_grader._add_glow`, `_grader.fade_toe`,
`_grader.highlight_drift`, `_grader.airy_haze`, `_grader.clarity_split`,
`grain.apply_film_grain`, `_grader.negative_split_tone`,
`_grader.channel_mixer_bw`.

Remaining uint8 boundaries in the global pipeline (ops that still
require uint8 and round-trip through `to_uint8`/`to_float`):
- `subject_aware_transfer` (`engine.py:2596`)
- `tonal.apply_hd_curve` (uses `cv2.LUT`, `engine.py:2640`)
- `_grader.grade_stack` (`engine.py:2647`)
- `add_impact_finish` (`engine.py:2811`)
- `_stage_body_skin` ops that call `blend_masked` (which expects uint8
  `[0,255]`, `engine.py:2371-2373`)

The float-native `_F_*` helpers (`_F_adjust_contrast`,
`_F_adjust_tonal`, `_F_adjust_vibrance`, `_F_apply_uniform_saturation`,
`_F_apply_selective_sharpening`) at `engine.py:2871-3035` are the
float32 `[0,1]` variants of the uint8 tonal operators.

## 4. Module map

All pipeline-stage modules live in `retouch/`. Entry points (`gui.py`,
`gui_advanced.py`, `gui_shoot.py`, `gui_batch.py`, `cli.py`) are thin
per `AGENTS.md`.

### Engine & orchestration

| Module | Role |
|--------|------|
| `engine.py` | `RetouchEngine`, `ProcessingContext`, `ProcessingResult`, `_CoreResult`, `build_context`, stage methods, `_F_*` float helpers. The orchestrator — no pixel math here. |
| `params.py` | `ParamSpec` registry (`PROCESSING_PARAMS`), `resolve_recipe`, `build_context` helpers, recipe→GUI→engine value translation. Single source of truth for every tunable. |
| `recipes.py` | Recipe dict table (`RECIPES`). Supports `extends` single-level inheritance. |
| `recipe_loader.py` / `recipe_loader_cli.py` | User-imported recipes from `~/.cache/retouch/user_recipes/`. |
| `recipe_schema.py` | Recipe validation (mandatory before pipeline entry per `AGENTS.md`). |
| `precision.py` | `to_float` / `to_uint8` / `ensure_float` / `PrecisionContext`. |
| `perf_optimizations.py` | `_process_face_core`, `FaceProcessorPool`, `_FaceResult`, `_norm_mask`/`_accum`/`_build_smooth_mask` helpers, `build_ort_providers`. Module-level so worker processes can import without circular deps. |
| `utils.py` | `correct_exposure`, `apply_global_bloom`, `apply_skin_diffusion`, `vibrance`, `squeeze_mask`, `feather_mask`, `guided_filter`, `normalize_mask`, `blend_masked`, `bgr_f32_to_lab_f32`/`lab_f32_to_bgr_f32`, `inter_eye_distance`. |
| `color_space.py` | `bgr_to_lch`, `skin_mask_lch`. |
| `color_science.py` | `bgr_to_oklab`, `oklab_to_oklch`. |

### Pipeline stages

| Stage | Module(s) | Class/function |
|-------|-----------|----------------|
| 0 Detection | `detection.py` | `FaceDetector`, `FaceData`, `FaceContext`, `_LandmarkCompat` |
| 0 Parsing | `parsing.py` | `FaceParser`, `FaceRegions` (28-region `__slots__` object) |
| 1 Reshape | `geometry.py` | `FaceReshaper` |
| 2 Frequency | `frequency.py` | `FrequencySeparator`, `FrequencyLayers`, `_texture_adaptation_factor` |
| 2 Skin | `skin.py` | `SkinProcessor` (smooth/flatten/restore_micro_texture/micro_dodge_burn/redness_even/equalize/unify_hue_line/unify_tone/whiten/shine_removal/quantize_tones/apply_specular_bloom/dodge_burn/wrinkle_soften/texture_transplant/local_clarity/harmonize_neck) |
| 2 Blemish | `blemish.py` | `BlemishRemover` |
| 2 Undereye | `undereye.py` | `UnderEyeRepairer` |
| 2 Eyes | `eyes.py` | `EyeEnhancer` |
| 2 Lips | `lips.py` | `LipEnhancer` |
| 2 Teeth | `teeth.py` | `TeethWhitener` |
| 2 Makeup | `makeup.py` | `MakeupEngine` (blush) |
| 2 Hair | `hair.py` | `HairEnhancer` |
| 2 Relight | `relight.py` | `Relighter` (face, landmark-based) |
| 2 Shadow lift | `shadow_lift.py` | `ShadowLifter` |
| 2 Heal (F4) | `heal.py` | `heal_region`, `b64_to_mask` (pre-pipeline) |
| 3 Subject sep | (inline in `engine.py`) | `_stage_subject_separation` |
| 3.5 Body skin | (inline in `engine.py`) + `body_relight.py`, `shadow_lift.py` | `_stage_body_skin`, `BodyRelighter` |
| 4 Tonal | (inline in `engine.py`) + `tonal.py` | `_stage_global`, `tonal.apply_hd_curve` |
| 5 Grading | `grading.py` | `ColorGrader`, `PRESETS`, `subject_aware_transfer` (in `style_transfer.py`) |
| 5 Highlight | `highlight.py` | `apply_highlight_rolloff` |
| 5 Grain | `grain.py` | `apply_film_grain` |
| 5 LUT | `lut.py` | LUT application |
| 5 Skin protect | `skin_protect.py` | skin-mask protection during grading |
| 6 Finish | (inline in `engine.py`) | `_stage_finish` (sharpen + `add_impact_finish`) |
| QA | `qa_detectors.py` | `run_all`, `QAWarning`, banding/clipping/plastic-skin/halo/seam detectors |

### Other

| Module | Role |
|--------|------|
| `style.py` / `style_library.py` / `style_transfer.py` | `StyleProfile` learning, dataset style extraction, subject-aware color transfer. |
| `advanced_contract.py` | Provenance/delivery truth contracts for Advanced Retouch output. |
| `shoot_review.py` + `watch_folder.py` | Shoot/review workflow + watch-folder state machine. |
| `harmony.py` | Face-anchored body mask (tone harmony between face and body). |
| `io.py` | `imread_exif`, export resolution/format maps. |
| `model_fetch.py` | ONNX/TFLite model download. |
| `regions.py` | Region-mask geometry helpers. |
| `hairwork.py` | Hair retouching helpers (distinct from `hair.py` enhancement). |
| `batch_processor.py` | Batch orchestration over `RetouchEngine`. |

## 5. Key dataclasses

### `ProcessingContext` (`engine.py:167`)

Fully-resolved, typed parameter bag for one `process()` call. ~120
fields grouped: skin, eyes, lips, teeth, makeup, hair, tonal, color
grading, split toning, white balance/B&W, bloom/lens, finish, subject
separation, internal recipe tag, modular flags, `quality`, and the
`face_contexts` cache.

Built by `build_context` (`engine.py:379`) from `resolve_recipe(name)`
+ caller overrides. Field defaults come from `_DEFAULTS` (built from
`PROCESSING_PARAMS` at `engine.py:163`) so the dataclass and the
`ParamSpec` registry agree.

Notable non-parameter fields:
- `quality: str = "full"` — selects F8.2 vs F8.1 (`engine.py:313`).
- `face_contexts: Optional[List[FaceContext]]` — cached detection +
  parsing results; `None` ⇒ engine detects/parses (`engine.py:316`).
- `heals: Optional[List[Dict]]` — F4 manual heal marks
  (`engine.py:319`).
- `active_recipe: str` — used for conditional logic inside stages.

### `_CoreResult` (`engine.py:583`)

Output of the core pipeline (stages 0–6). Carried between phases:

```python
@dataclass
class _CoreResult:
    result: np.ndarray
    acc_skin: Optional[np.ndarray]
    acc_skin_hair: Optional[np.ndarray]
    acc_lips: Optional[np.ndarray]
    acc_sharpen: Optional[np.ndarray]
    faces: list
    person_mask: Optional[np.ndarray]
    no_face: bool = False
    face_contexts: Optional[List["FaceContext"]] = None
    qa: List[QAWarning] = field(default_factory=list)
```

### `FaceContext` (`detection.py:48`)

Cached per-face detection + parsing results. Picklable for ProcessPool
IPC.

```python
@dataclass
class FaceContext:
    face_data: FaceData
    regions: FaceRegions
    index: int = 0
    face_image: Optional[np.ndarray] = None
```

### `FaceData` (`detection.py:39`)

```python
@dataclass
class FaceData:
    landmarks: object               # _LandmarkCompat (478 NormalizedLandmark)
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    ied: float                      # inter-eye distance, pixels
    confidence: float = 1.0
```

### `FaceRegions` (`parsing.py:109`)

`__slots__` object holding 28 per-region float32 masks for one face
(skin, forehead, cheeks, nose, eyes, eyebrows, irises, lips, mouth
interior, under-eyes, nose bridge, forehead center, cheek highlights,
jawline contour, hair, neck, nasolabial folds, crow's feet, face
oval). Populated by `FaceParser.parse` from BiSeNet ONNX + landmark
geometry.

### `_FaceResult` (`perf_optimizations.py:165`)

Per-face output from `_process_face_core`:

```python
@dataclass
class _FaceResult:
    canvas: np.ndarray
    skin_mask: np.ndarray
    skin_hair_mask: np.ndarray
    lips_mask: np.ndarray
    sharpen_mask: np.ndarray
    roi_box: Tuple[int, int, int, int]
```

### `ProcessingResult` (`engine.py:326`)

Public return type of `process()`. Subclasses `np.ndarray` so legacy
callers can treat it as a bare uint8 BGR array, while also carrying
`skin_mask`, `skin_hair_mask`, `sharpen_mask`, `lips_mask`,
`face_count`, `params`, `timings`, `face_contexts`, `qa`.

## 6. Threading model

### Gradio (`gui.py`)

- One global `RetouchEngine` instance, lazily built under
  `_engine_lock = threading.Lock()` (`gui.py:46`).
- Gradio launched with `app.queue(default_concurrency_limit=1)`
  (`gui.py:1747`) — serialises requests, so the engine instance is
  not accessed concurrently from the GUI path. The lock + singleton
  guard is defence-in-depth.
- `AGENTS.md` requires engine instances and `FaceContext` caches to be
  thread-safe for Gradio; the `default_concurrency_limit=1` queue is
  the primary safety mechanism.

### Face processing pool (`perf_optimizations.py:751`)

`FaceProcessorPool` wraps a `ProcessPoolExecutor` with the `spawn`
multiprocessing context (`perf_optimizations.py:765`).

- Lazily started on first multi-face call; shut down in
  `RetouchEngine.close()` (`engine.py:2852`).
- Default `max_workers=4` (`perf_optimizations.py:766`).
- **Single face bypasses IPC** — processed inline in the parent process
  (`perf_optimizations.py:808`).
- Multi-face: each face is submitted as a picklable payload tuple
  (`engine.py:1935-1949`) to `_process_single_face_worker`
  (`perf_optimizations.py:690`).
- Worker processors (`SkinProcessor`, `BlemishRemover`, etc.) are
  **lazily built and cached per worker process** in
  `_WORKER_PROCESSORS` (`perf_optimizations.py:660`). These are cheap
  (no ONNX); only `FaceParser` loads ONNX and region masks are
  pre-computed in the parent, so no ONNX session crosses the process
  boundary.
- `ProcessingContext` is pickled as a **slimmed dict** (ndarray fields
  stripped, `engine.py:1930`) to minimise IPC payload. The worker
  reconstructs it as a `SimpleNamespace` (`perf_optimizations.py:301`)
  — **this is the multiprocess pickling pitfall called out in
  `AGENTS.md`**: `ctx` arrives as a `dict` but is accessed via
  attribute syntax. Must convert with `SimpleNamespace(**ctx)` in the
  worker before use.
- On worker failure: returns `None` for that face, parent falls back to
  in-process `_process_one_face` (`engine.py:1956`).
- On `FaceProcessorPool` failure: falls back to `ThreadPoolExecutor`
  with `max_workers=min(len(faces), 4)` (`engine.py:1974`), sharing the
  engine instance via memory (GIL-released OpenCV/NumPy ops).

## 7. Model dependencies

All models live in `models/` (resolved relative to the package root,
`detection.py:25`).

| Model | File | Format | Used by | Purpose |
|-------|------|--------|---------|---------|
| MediaPipe FaceLandmarker | `models/face_landmarker.task` | MediaPipe Tasks | `FaceDetector` (`detection.py:26`) | 478-point face landmarks. Required — `FaceDetector.__init__` raises `FileNotFoundError` if missing (`detection.py:85`). |
| MediaPipe Selfie Segmenter | `models/selfie_segmenter.tflite` | TFLite | `FaceDetector.segment_person` (`detection.py:27`) | Person/background mask. Optional — `segment_person` returns all-ones mask if `_segmenter is None` (`detection.py:273`). |
| BiSeNet (ResNet18) face parsing | `models/resnet18.onnx` | ONNX | `FaceParser` (`parsing.py:137`) | 19-class face region segmentation (skin, eyes, brows, lips, neck, hair, …). Optional — `FaceParser` falls back to landmark-only regions if the session fails to init (`parsing.py:140`). |
| RetinaFace (pip) | — | pip package `retinaface` | `FaceDetector.detect` (`detection.py:169`) | Bounding-box detection. Optional — falls back to MediaPipe FaceLandmarker on full image if unavailable. |
| RetinaFace MV1 | `models/retinaface_mv1.onnx` | ONNX | (present in `models/` but not loaded by current `detection.py`) | Reserved/legacy. |

### ONNX execution providers (`perf_optimizations.py:843`)

`build_ort_providers()` auto-discovers providers in priority order:
1. **CoreMLExecutionProvider** (Apple Silicon: Neural Engine + GPU,
   `MLProgram` format, `ALL` compute units) — `perf_optimizations.py:852`
2. **CUDAExecutionProvider** (NVIDIA, 2GB GPU mem cap, cuDNN
   `EXHAUSTIVE` algo search) — `perf_optimizations.py:866`
3. **DmlExecutionProvider** (Windows DirectML: AMD/Intel/Qualcomm) —
   `perf_optimizations.py:879`
4. **CPUExecutionProvider** (always appended as fallback) —
   `perf_optimizations.py:888`

Used by `FaceParser.__init__` (`parsing.py:146`). If provider
initialisation fails, falls back to default `ort.InferenceSession`
(`parsing.py:152`).

### Numba (optional, `perf_optimizations.py:73`)

`HAS_NUMBA` gates JIT-compiled pixel loops with `prange` parallelism.
If Numba is missing, a `DummyNumbaModule` provides no-op `jit`/`prange`
fall-backs (`perf_optimizations.py:83`). Engine startup attempts to
warm up JIT kernels (`engine.py:650` comment).

---

## P3 refactor notes

The current pipeline is method-based on a single `RetouchEngine` class:
stages are private methods (`_stage_reshape`, `_stage_per_face`,
`_stage_subject_separation`, `_stage_body_skin`, `_stage_global`,
`_stage_grade`, `_stage_finish`) plus the F8.1/F8.2 phase split
(`_run_detection_and_faces`, `_process_native_faces`,
`_run_global_phases`). The per-face sub-chain inside
`_process_face_core` is a flat sequence of `if ctx.X > 0:` blocks, not
a registry.

For the P3 stage-registry refactor, the natural seams are:

1. **Per-face sub-ops** in `_process_face_core`
   (`perf_optimizations.py:228-649`) — each `if ctx.X > 0:` block is a
   candidate stage entry. The `_E1_CHAIN_ORDER` list
   (`perf_optimizations.py:205`) already names them in execution order.
2. **Global stages** in `_run_global_phases` (`engine.py:1520`) —
   stages 3, 3.5, 4, 5, 6 are already method-isolated.
3. **Resolution strategy** (`_process_with_proxy`, `_process_native_faces`,
   `_run_detection_and_faces`) — the F8.1/F8.2 split is the registry's
   first-class dispatch point.
4. **`_CoreResult`** (`engine.py:582`) is already the inter-stage data
   contract; a registry should formalise it as the stage I/O type.
5. **`ProcessingContext`** (`engine.py:167`) is the parameter bag; a
   registry should let each stage declare which `ctx.*` fields it
   consumes (the `_ENGINE_QUIRK_CONVERSIONS` set at `engine.py:467`
   already hints at this mapping).
