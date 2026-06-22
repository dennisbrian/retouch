# Architecture: Retouch Pipeline

## Stages 0–6

| Stage | Module | Function |
|---|---|---|
| 0 | Detection & Segmentation | RetinaFace detection, MediaPipe selfie mask, exposure correction; `FaceContext` cache bypasses on re-run |
| 1 | Face Reshaping | Liquid warp on jaw/cheek/chim landmarks; single `cv2.remap` per image |
| 2 | Per-Face Processing | BiSeNet ONNX mask → 3 freq bands (bilateral float32); eyes/lips/teeth/blemish/blush/relight; blend back |
| 3 | Global Tonal | Parametric curves: exposure, contrast, highlights, shadows, whites, blacks |
| 4 | Subject-Background | Selfie mask drives distinct subject vs bg adjustments |
| 5 | Color Grading | Color transfer, split-toning, Orton bloom, grain, CA, halation, vignette |
| 6 | Sharpening | Unsharp mask on face/hair/eyebrow edges; final impact curves |

## High-Res Proxy

Canvas >2048px → downscale to `PROXY_MAX_DIM=2048` → run pipeline → `cv2.INTER_LINEAR` upscale masks/result. Keeps peak RAM <2GB.

## Model Backends

- **BiSeNet ONNX**: `build_ort_providers` (prefers `CoreMLExecutionProvider` on macOS, fallback `CPUExecutionProvider`)
- **MediaPipe**: explicit `base.Delegate.CPU` to prevent thread hangs
