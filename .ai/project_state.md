# Project State: Retouch v2.0.0

Python 3.9+ · OpenCV, MediaPipe, ONNX Runtime, NumPy, SciPy, Gradio

## Entry Points

| File | Role |
|---|---|
| `cli.py` | Batch CLI engine |
| `gui.py` | Gradio web dashboard |
| `desktop.py` | Desktop wrapper |

## Module Map

| Module | Responsibility |
|---|---|
| `engine.py` | Pipeline orchestration, proxy scaling, multi-face scheduling |
| `detection.py` | Face detector, selfie segmenter, coord fitting |
| `geometry.py` | FaceReshaper: jaw/cheek/chim liquid warp |
| `parsing.py` | BiSeNet ONNX → per-region facial masks |
| `frequency.py` | 3-level freq separation (coarse/mid/fine) + pore synthesis |
| `skin.py` | Whitening, CLAHE, Dodge & Burn, neck tone match |
| `lips.py` / `eyes.py` / `teeth.py` | Gloss/matte/velvet, sclera/iris contrast, whitening |
| `hair.py` / `makeup.py` / `undereye.py` / `blemish.py` | Hair shine, blush, dark circle repair, inpainting |
| `relight.py` | Blinn-Phong 3D shading via Delaunay depth maps |
| `grading.py` | LUT, split-toning, bloom, CA, halation |
| `style.py` / `style_library.py` | Style profiles, Reinhard transfer, dataset learning |
| `batch_processor.py` | Folder ingestion, classification cache, contact sheets |
| `utils.py` | Mask feather, blending, math helpers |
| `io.py` | EXIF preservation, RAW decoding (rawpy) |

## State

- **Proxy scaling**: stable
- **FaceContext cache**: bypasses detection/parsing on slider updates
- **Parallelization**: subprocess pool for ≥2 faces, thread fallback

## Known Limits

- ~4% detection failure on extreme profiles / high-contrast occlusion
- ONNX `CoreMLExecutionProvider` unavailable on some Apple Silicon — falls back to CPU
