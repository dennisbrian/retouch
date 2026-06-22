# Architecture: Retouch Pipeline

## Stages 0–6

| Stage | Module | Function |
|---|---|---|
| 0 | Detection & Segmentation | RetinaFace detection, MediaPipe selfie mask, exposure correction; `FaceContext` cache bypasses on re-run |
| 1 | Face Reshaping | Liquid warp on jaw/cheek/chim landmarks; single `cv2.remap` per image |
| 2 | Per-Face Processing | BiSeNet ONNX mask → 3 freq bands (`FrequencySeparator.separate`/`combine`); eyes/lips/teeth/blemish/blush/relight; blend back |
| 3 | Global Tonal | Parametric curves: exposure, contrast, highlights, shadows, whites, blacks |
| 4 | Subject-Background | Selfie mask drives distinct subject vs bg adjustments |
| 5 | Color Grading | Color transfer, split-toning, glow + vignette, Orton bloom, grain, CA, halation |
| 6 | Sharpening | Unsharp mask on face/hair/eyebrow edges; final impact curves |

## High-Res Proxy & Fast Preview

Two internal resolution layers (output file dimensions always preserved when `export_res=Original`):

- **Fast Preview** (`fast=True`, GUI default): downscale to 800px → run pipeline → upscale. ~3-5× faster.
- **Proxy Resolution** (auto for inputs >2048px): downscale to `PROXY_MAX_DIM=2048` → run pipeline → upscale masks/result. Keeps peak RAM <2GB.

For high-res production: keep `fast=True` for tuning, turn off for final export. The per-face work (BiSeNet, freq separation) runs at the reduced resolution, so fine details may be slightly softer when `fast=True` or input > 2048px.

## Central Parameter Registry (`retouch/params.py`)

`ParamSpec` dataclass + `PROCESSING_PARAMS` list (59 entries) is the single source of truth for all tunable parameters. Used by:
- `engine.build_context()` — recipe → ProcessingContext translation
- `gui.recipe_defaults()` — recipe → GUI slider values
- `cli.build_params()` — CLI args → engine kwargs

Adding a new param = one entry. Solves the 7-way sync problem.

## Model Backends

- **BiSeNet ONNX**: `build_ort_providers` (prefers `CoreMLExecutionProvider` on macOS, fallback `CPUExecutionProvider`)
- **MediaPipe**: explicit `base.Delegate.CPU` to prevent thread hangs

## Circular Dependencies — NONE

- `engine → style → engine` broken: `style.py` imports `FaceDetector`/`FaceParser` directly
- `engine → parsing → perf_optimizations → engine` broken: picklable helpers (`_process_face_core`, `_norm_mask`, `_FaceResult`, `_accum`) moved to `perf_optimizations.py`
