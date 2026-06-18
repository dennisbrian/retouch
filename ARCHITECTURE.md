# Pro Max Face Retouch Engine — Architecture & Pipeline Design

This document details the architectural design, processing pipeline, and component breakdown of the **Pro Max Face Retouch Engine (v2.x)**.

---

## 1. Dual-Engine Context

The workspace contains two co-existing versions of the engine:

1.  **Legacy Engine (`/Applications/htdocs/retouch/engine.py` at root)**:
    *   A simple, single-file prototype.
    *   Relies entirely on MediaPipe landmark indices to build primitive polygon masks for face regions, resulting in hard, unnatural transitions.
    *   Only handles basic bilateral frequency smoothing and static LAB color shifts. No advanced feature processing or color grading.
2.  **Production Engine (`/Applications/htdocs/retouch/retouch/`)**:
    *   A professional-grade, modular pipeline.
    *   Combines **deep-learning semantic segmentation (BiSeNet ONNX)** with 3D landmark mesh mapping.
    *   Features a 15-stage processing pipeline including blemish inpainting, face-to-neck matching, specular lip gloss, hair shine, Dodge & Burn, and LAB-space color grading.

---

## 2. Global Pipeline Architecture

The workflow is divided into: **Face Detection** (bounding boxes), **Crop Landmark Fitting** (3D mesh coordinates), **Semantic Region Parsing** (pixel-precise masks), **Frequency Separation** (texture isolation), **Targeted Enhancement** (component-level edits), and **Global Finishing** (grading, vignettes, and bloom).

![Pipeline Architecture](pipeline_architecture.png)

```mermaid
graph TD
    %% Define Styles and Colors
    classDef inputOutput fill:#efebe9,stroke:#8d6e63,stroke-width:2px,color:#3e2723;
    classDef detection fill:#ede7f6,stroke:#7e57c2,stroke-width:2px,color:#1a237e;
    classDef segmentation fill:#e0f2f1,stroke:#26a69a,stroke-width:2px,color:#004d40;
    classDef frequency fill:#fbe9e7,stroke:#ff5722,stroke-width:2px,color:#bf360c;
    classDef enhancement fill:#fff8e1,stroke:#ffca28,stroke-width:2px,color:#5d4037;
    classDef finishing fill:#e3f2fd,stroke:#42a5f5,stroke-width:2px,color:#0d47a1;
    classDef model fill:#fafafa,stroke:#bdbdbd,stroke-width:1px,stroke-dasharray: 5 5,color:#616161;

    %% Nodes
    input["Input Image (BGR)"]:::inputOutput
    
    subgraph Stage1["Stage 1: Face Detection & Landmarking"]
        detector["RetinaFace detector<br/><small>Bounding boxes · BGR→RGB fix · TF_USE_LEGACY_KERAS env var</small>"]:::detection
        crop["Crop + 30% pad<br/><small>FaceLandmarker on crop</small>"]:::detection
        remap["Remap to full coords"]:::detection
        fallback["Full image fallback<br/><small>FaceLandmarker on full frame</small>"]:::detection
        facedata["FaceData collection<br/><small>478 landmarks · IED · bbox · score</small>"]:::detection
    end

    subgraph Stage2["Stage 2: Semantic Segmentation"]
        selfie["Selfie segmenter<br/><small>Person vs. background mask</small>"]:::segmentation
        bisenet["BiSeNet ONNX<br/><small>Pixel-precise semantic masks</small>"]:::segmentation
        faceregions["FaceRegions masks<br/><small>Skin · brows · eyes · lips · neck · hair · IED feathering</small>"]:::segmentation
    end

    subgraph Stage3["Stage 3: Bilateral Smoothing"]
        freqsep["3-level frequency separation<br/><small>Coarse · medium · fine bands · Gaussian radius scaled to face width</small>"]:::frequency
        smoothing["Bilateral skin smoothing<br/><small>Preserves pores · peach fuzz · avoids plastic look</small>"]:::frequency
    end

    subgraph Stage4_9["Stages 4-9: Component-Level Enhancements"]
        blemish["Blemish removal<br/><small>Fast marching inpaint</small>"]:::enhancement
        eye["Eye enhancement<br/><small>Sclera · iris · catchlight</small>"]:::enhancement
        teeth["Teeth whitening<br/><small>LAB/HSV yellow removal</small>"]:::enhancement
        lip["Lip gloss<br/><small>Saturation + specular</small>"]:::enhancement
        hair["Hair shine<br/><small>Structural highlight boost</small>"]:::enhancement
        sculpt["Sculpting<br/><small>Under-eye · Dodge & Burn</small>"]:::enhancement
        neck["Neck matching & skin equalisation"]:::enhancement
    end

    subgraph Stage10_12["Stages 10-12: Global Finishing"]
        grading["Global colour grading<br/><small>LAB S-curves · split toning shadows/highlights · clarity</small>"]:::finishing
        vignette["Vignette & micro-contrast"]:::finishing
    end

    subgraph Stage13["Stage 13: Glow Layer"]
        bloom["Smart highlight bloom<br/><small>Screen blur · L > 180 restricted · dreamy glow</small>"]:::finishing
    end

    output["Output Image (BGR)"]:::inputOutput

    %% Model definitions
    subgraph Models["Model Assets"]
        m1["face_landmarker.task<br/><small>MediaPipe</small>"]:::model
        m2["resnet18.onnx (BiSeNet)<br/><small>CoreML → CPU fallback</small>"]:::model
        m3["selfie_segmenter.tflite<br/><small>TFLite</small>"]:::model
    end

    %% Connections
    input --> detector
    detector -->|primary| crop
    detector -->|fallback| fallback
    crop --> remap
    remap --> facedata
    fallback --> facedata
    
    facedata --> selfie
    facedata --> bisenet
    selfie --> faceregions
    bisenet --> faceregions
    
    faceregions --> freqsep
    freqsep --> smoothing
    
    smoothing --> blemish
    smoothing --> eye
    smoothing --> teeth
    smoothing --> lip
    smoothing --> hair
    smoothing --> sculpt
    
    blemish --> neck
    eye --> neck
    teeth --> neck
    lip --> neck
    hair --> neck
    sculpt --> neck
    
    neck --> grading
    grading --> vignette
    vignette --> bloom
    bloom --> output
```

---

## 3. Core Modules & Stage Breakdown

### 3.1. Face Detection & Landmarking (`retouch/detection.py`)
*   **RetinaFace (Primary)**: Detects face bounding boxes.
    *   *Keras 3 Runtime Fix*: Sets `os.environ["TF_USE_LEGACY_KERAS"] = "1"` at initialization. This avoids silent execution crashes due to symbolic tensor serialization errors under Keras 3.x / TensorFlow 2.16+.
*   **Crop-based Fitting**: Crops face regions with 30% padding and runs `FaceLandmarker` on the crop. Landmarks are remapped back to full-image coordinates. This makes landmark fitting highly robust on side profiles and distant subjects.
*   **MediaPipe Landmarker (Fallback)**: If RetinaFace fails or is unavailable, the system runs landmarker on the full image.

### 3.2. Face Reshaping / Slimming (`retouch/geometry.py`)
*   **FaceReshaper**: Applies photographer-grade local translation warping (liquid warping) to jawline landmarks 234 & 454 (inward shift of 2%–5%), cheeks 117 & 346 (inward shift of 1%–3%), and chin 152 (upward shift of 1%–2%).
*   **Parallel Execution Grid**: Accumulates coordinate displacements for all faces first, executing a single `cv2.remap` for zero-overhead performance.

### 3.3. Makeup Engine (`retouch/makeup.py` & `retouch/lips.py`)
*   **Blush Engine**: Generates heavily feathered radial masks around cheek landmarks 117 & 346, applying a rosy flush by nudging the LAB `a` channel positively.
*   **Lipstick Finishes**: Supports `matte` (pure color tint preserving lip textures), `gloss` (adds specular gloss reflection highlights), and `velvet` (bilateral smoothing on lip textures and reduced opacity texture overlay).

### 3.4. Face Parsing & Region Masking (`retouch/parsing.py`)
*   **BiSeNet ResNet18 ONNX**: Performs pixel-precise semantic segmentation of face parts. Output categories (skin, eyebrows, eyes, mouth interior, lips, neck, hair) are converted to float32 masks.
*   **Adaptive Feathering**: Feathers mask edges dynamically based on the **Inter-Eye Distance (IED)** to guarantee seamless blending during skin adjustments.
*   **Landmark Fallback**: If ONNX inference is bypassed, it generates polygon-based region masks using specific MediaPipe landmarks.

### 3.5. Frequency Separation & Smoothing (`retouch/frequency.py` & `retouch/skin.py`)
*   **3-Level Separation**: Splits the image into coarse (color/tonal flow), medium (minor skin structures), and fine (pore details/hair strands) frequency bands using Gaussian blur radius scaled to the face width.
*   **Bilateral Filtering**: Smooths low/medium bands to level out skin blotchiness while conserving high-frequency details (pores, peach fuzz), keeping the texture natural and preventing a "plastic" look.

### 3.6. Component-Level Enhancers
*   **Blemish Removal (`retouch/blemish.py`)**: Identifies high-frequency blemishes on the skin mask and applies fast marching inpainting.
*   **Eyes (`retouch/eyes.py`)**: Whitens the sclera (using LAB luminance boosts) and sharpens/saturates the iris. Boosts catchlights and reflections by up to 25%.
*   **Teeth (`retouch/teeth.py`)**: Segments the mouth interior and whitens/desaturates yellow-hued pixels in the LAB/HSV color space.
*   **Lips (`retouch/lips.py`)**: Saturation/vibrance boost combined with specular highlight isolation to simulate lip gloss.
*   **Hair & Outfit (`retouch/hair.py`)**: Segments hair and outfit boundaries using the selfie segmenter and applies structural highlight boosts.
*   **Under-Eye & Sculpting (`retouch/undereye.py` & `retouch/skin.py`)**: Lightens dark circles and applies subtle Dodge & Burn contours to the nose bridge, forehead center, cheeks, and jawline.

### 3.7. Color Grading & Finishing (`retouch/grading.py`)
*   **Luminance Curves**: S-curves applied to the LAB L-channel to shape contrast.
*   **Split Toning**: Colorizes shadows and highlights independently using target LAB $a/b$ vectors.
*   **Smart Glow (Soft Bloom)**: Screens a heavily blurred version of the image back onto itself, restricted strictly to high-brightness areas ($L > 180$) to simulate dreamy lighting without washing out shadows.
*   **Vignetting & Clarity**: Radial falloff vignettes and unsharp mask-based micro-contrast.

---

## 4. Model Configurations & Execution Providers

ONNX models are located in `models/` and initialized with hardware acceleration options:
*   **resnet18.onnx** (BiSeNet): Prefers `CoreMLExecutionProvider` on Apple Silicon macOS, falling back automatically to `CPUExecutionProvider` on other architectures.
*   **face_landmarker.task**: MediaPipe task binary.
*   **selfie_segmenter.tflite**: TensorFlow Lite segmenter.

---

## 5. Detection Benchmarks & Postmortem

Following a batch analysis of 699 frames, the face detection subsystem was optimized to address a series of silent failures and edge cases.

### Detection Performance (Before vs. After)

| Metric | Before | After |
| :--- | :--- | :--- |
| **Detection Rate** | 39% (274/699) | **~96% (est. 670+/699)** |
| **Per-Image Time (2048px)** | ~500ms | ~800ms |

> [!NOTE]
> The per-image latency increase from ~500ms to ~800ms occurs because the detection success rate rose from 39% to 96%. This means the "fast-reject" path (where no face is found and minimal processing is done) is rarely triggered. In production pipelines using `--workers 6`, this extra 300ms is fully absorbed by parallel retouch stages.

### Root Cause Analysis & Optimization Summary

| # | Optimization | Impact / Finding |
| :--- | :--- | :--- |
| **1** | **Keras 3 Runtime Fix** | Added `os.environ["TF_USE_LEGACY_KERAS"] = "1"` to bypass Keras 3.x / TensorFlow 2.16+ symbolic tensor initialization exceptions that were causing RetinaFace to crash silently. |
| **2** | **Per-Face Fallback Pass** | Symmetrical fallback mechanism: if RetinaFace detects boxes but the crop-based landmarking fails (due to angles or occlusion), the system triggers full-image MediaPipe detection and deduplicates results. This was the primary driver of the recall boost. |
| **3** | **Symmetrical Padding** | Replaced asymmetric coordinate clipping near image borders with symmetrical reduced padding to prevent face off-centering during crops, resolving landmarking failures. |
| **4** | **Exception logging** | Switched from blank `except Exception` blocks to logging warnings for non-import errors, making future failures transparent. |
| **5** | **Confidence threshold** | Kept at `0.5` as it was not the bottleneck. |

### Remaining Edge Cases (~4%)
The remaining ~4% of undetected frames (e.g., `DSCF6900`) represent extreme profiles, high-contrast shadow occlusion, or tiny faces in distant environment shots where MediaPipe landmarks cannot be mathematically resolved.

