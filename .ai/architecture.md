# Architecture: Pro Max Face Retouch Pipeline

## Pipeline Pipeline (Stages 0–6)

The engine decomposes image processing into seven sequential stages:

### Stage 0: Detection & Segmentation
- Runs face detection (RetinaFace primary, MediaPipe landmarker fallback).
- Computes person/background mask via MediaPipe Selfie Segmenter.
- Applies optional exposure correction using face histogram bounds.
- Supports `FaceContext` caching (stores landmarker outputs and BiSeNet masks to skip detection on sub-sequent runs).

### Stage 1: Face Reshaping
- Applies local liquid warp displacements to jaw, cheek, and chin landmarks.
- Accumulates displacements across all faces before executing a single `cv2.remap` call.

### Stage 2: Per-Face Processing
- Crops face ROI (expanded bounding box: +80% top, +180% bottom, +60% sides).
- Runs BiSeNet ONNX segmentation to produce region masks (skin, hair, eyebrows, eyes, nose, lips).
- Feathers masks based on Inter-Eye Distance (IED).
- Splits ROI into 3 frequency bands (coarse, mid, fine).
- Smooths coarse/mid bands (bilateral filtering on float32) while preserving fine details.
- Runs local enhancements (eyes, lips, teeth, blemish inpainting, radial blush, dark circles).
- Performs Blinn-Phong 3D shading on MediaPipe depth maps (relighting).
- Blends the processed ROI back onto the canvas.

### Stage 3: Global Tonal Adjustments
- Adjusts global exposure, contrast, highlights, shadows, whites, and blacks via parametric curves.

### Stage 4: Subject-Background Separation
- Uses the Stage 0 selfie segmenter mask to apply distinct adjustments to the subject vs background.

### Stage 5: Color Grading & Post-Effects
- Applies color transfer from reference image or runs split-toning grading presets.
- Processes Orton bloom, film grain, chromatic aberration, halation, and vignetting.

### Stage 6: Sharpening & Impact Finish
- Applies selective unsharp masking to face, hair, and eyebrow edges.
- Runs final contrast and saturation impact curves.

## High-Resolution Proxy Optimization
- For canvases exceeding 2048px (e.g. 24 MP):
  1. Downscales canvas to a 2048px proxy (`PROXY_MAX_DIM`).
  2. Runs detection, parsing, and pipeline calculations on the proxy.
  3. Upscales result and generated blending masks to original size via `cv2.INTER_LINEAR`.
  4. Blends output using high-resolution masks.
- Prevents out-of-memory errors (limits peak RAM usage to < 2 GB).

## Model Run Options
- **BiSeNet ONNX**: Uses `build_ort_providers` (prefers `CoreMLExecutionProvider` on macOS, falls back to `CPUExecutionProvider`).
- **MediaPipe elements**: Configured with explicit CPU delegation (`base.Delegate.CPU`) to prevent thread hangs/crashes across environments.
