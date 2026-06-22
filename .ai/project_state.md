# Project State: Pro Max Face Retouch Engine (v2.0.0)

## Codebase Metadata
- **Language**: Python 3.9+
- **Frameworks**: OpenCV, MediaPipe, ONNX Runtime, NumPy, SciPy, Gradio
- **Main Entry Points**:
  - [cli.py](file:///Applications/htdocs/retouch/cli.py) — Batch CLI engine execution
  - [gui.py](file:///Applications/htdocs/retouch/gui.py) — Interactive web dashboard (Gradio)
  - [desktop.py](file:///Applications/htdocs/retouch/desktop.py) — Desktop app wrapper

## Module Mapping & Responsibilities
- [engine.py](file:///Applications/htdocs/retouch/retouch/engine.py): Pipeline orchestrator (`RetouchEngine`), proxy scaling, multi-face scheduling.
- [detection.py](file:///Applications/htdocs/retouch/retouch/detection.py): `FaceDetector`, MediaPipe selfie segmentation, coordinate fitting.
- [geometry.py](file:///Applications/htdocs/retouch/retouch/geometry.py): `FaceReshaper` for liquid jaw/cheek slimming & chin lifts.
- [parsing.py](file:///Applications/htdocs/retouch/retouch/parsing.py): `FaceParser` running BiSeNet ONNX for pixel-precise facial masks.
- [frequency.py](file:///Applications/htdocs/retouch/retouch/frequency.py): 3-level frequency separation (coarse, mid, fine bands) & pore synthesis.
- [skin.py](file:///Applications/htdocs/retouch/retouch/skin.py): Skin whitening, CLAHE equalization, Dodge & Burn, neck tone matching.
- [lips.py](file:///Applications/htdocs/retouch/retouch/lips.py) / [eyes.py](file:///Applications/htdocs/retouch/retouch/eyes.py) / [teeth.py](file:///Applications/htdocs/retouch/retouch/teeth.py): Component enhancers (gloss/matte/velvet finishes, sclera/iris contrasts, teeth whitening).
- [hair.py](file:///Applications/htdocs/retouch/retouch/hair.py) / [makeup.py](file:///Applications/htdocs/retouch/retouch/makeup.py) / [undereye.py](file:///Applications/htdocs/retouch/retouch/undereye.py) / [blemish.py](file:///Applications/htdocs/retouch/retouch/blemish.py): Blemish inpainting, radial blush, under-eye repair, hair shine lifts.
- [relight.py](file:///Applications/htdocs/retouch/retouch/relight.py): 3D Blinn-Phong relighting using Delaunay depth maps.
- [grading.py](file:///Applications/htdocs/retouch/retouch/grading.py): LUT mapping, split toning, Orton bloom/glow, chromatic aberration, halation.
- [style.py](file:///Applications/htdocs/retouch/retouch/style.py) / [style_library.py](file:///Applications/htdocs/retouch/retouch/style_library.py): Style profiles, Reinhard color transfer, dataset learning from edited pairs.
- [batch_processor.py](file:///Applications/htdocs/retouch/retouch/batch_processor.py): Folder ingestion, image classification caching, contact sheet generation.
- [utils.py](file:///Applications/htdocs/retouch/retouch/utils.py): Common mask feathering, blending, and math helpers.
- [io.py](file:///Applications/htdocs/retouch/retouch/io.py): EXIF preservation, RAW camera decoding (via `rawpy`).

## Core Capability Status
- **Proxy Scaling**: Stable. High-res images (> 2048px) downscaled to 2048px proxy, processed, and upscaled.
- **Caching**: `FaceContext` caching bypasses detector/parser stages on slider updates.
- **Parallelization**: `FaceProcessorPool` subprocesses parallelize multi-face execution, falling back to a thread pool.

## Known Limitations
- **Detection Edge Cases**: ~4% failure rate in face detection on extreme profiles or high-contrast shadow occlusions where MediaPipe landmarker cannot locate points.
- **ONNX Execution Provider**: Bypasses GPUDevice on some Apple Silicon chips unless `CoreMLExecutionProvider` is correctly compiled; default is `CPUExecutionProvider` fallback.
