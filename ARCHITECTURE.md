# Pro Max Face Retouch Engine — Architecture & Pipeline Design

This document details the architectural design, processing pipeline, and component breakdown of the **Pro Max Face Retouch Engine (v3.0)**.

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
    *   Features a **17-stage processing pipeline** split into local face retouching and global image styling. Handles advanced features like blemish inpainting, face-to-neck matching, specular lip gloss finishes, hair shine lifts, Dodge & Burn, parametric tonal adjustments, reference-based color transfer, and stacked color grading.
    *   **High-Resolution ROI Optimization**: Processes high-resolution images (up to 24 MP and beyond) by extracting and processing a padded Portrait ROI enclosing the face, hair, and neck, reducing peak memory from **7.5 GB to 1.84 GB** and total runtime from **15.3s to 3.09s**.

---

## 2. Global Pipeline Architecture

The workflow is divided into: **Face Detection** (bounding boxes), **Portrait ROI Extraction & Landmark Fitting** (shifting coordinates relative to crop), **Semantic Region Parsing** (pixel-precise masks), **Frequency Separation** (texture isolation), **Targeted Enhancement** (component-level edits), **Seamless ROI Blend Back**, and **Global Finishing** (brightness, curves, reference color transfer, grading, vignettes, and bloom).

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
    
    subgraph Stage1_2["Stage 1-2: Detection, Slimming & Segmenter"]
        detector["Face detector<br/><small>RetinaFace / MediaPipe Fallback<br/>Explicit CPU delegation</small>"]:::detection
        slimming["Liquid face reshaping<br/><small>Slimming & chin lifts</small>"]:::detection
        person_mask["Selfie segmenter<br/><small>Person vs. background mask</small>"]:::segmentation
    end

    subgraph ROI_Crop["Portrait ROI Cropper (High-Res Optimization)"]
        roi_extract["Portrait ROI Crop<br/><small>Face + Neck + Hair bounding box</small>"]:::detection
        coord_shift["Landmark Coordinate Shifting<br/><small>Map landmarks relative to ROI crop</small>"]:::detection
    end

    subgraph Stage2_Parse["Stage 2: Semantic Segmentation (ROI Crop)"]
        bisenet["BiSeNet ONNX<br/><small>Pixel-precise parsing on crop</small>"]:::segmentation
        faceregions["FaceRegions masks<br/><small>Skin cleaning post-feathering<br/>Left/right under-eye masks</small>"]:::segmentation
    end

    subgraph Stage3_4["Stages 3-4: Frequency Separation & Smoothing (ROI Crop)"]
        freqsep["3-level frequency separation<br/><small>Gaussian blur scaled to face width</small>"]:::frequency
        smoothing["Frequency smoothing<br/><small>Smooth & mid_reduction controls<br/>Independent nose_smooth</small>"]:::frequency
    end

    subgraph Stage5_12["Stages 5-12: Component-Level Enhancements (ROI Crop)"]
        skin["Skin foundation<br/><small>Equalization & whitening<br/>Bi-directional support (tanning)<br/>Shadow protection (L > 80)</small>"]:::enhancement
        blemish["Blemish removal<br/><small>Fast marching inpaint</small>"]:::enhancement
        undereye["Under-eye repair<br/><small>Dark circles correction</small>"]:::enhancement
        neck["Neck matching<br/><small>Bi-directional face-to-neck skin blending</small>"]:::enhancement
        eyes_teeth["Eyes & Teeth<br/><small>Sclera, iris, catchlight, teeth whitening</small>"]:::enhancement
        lips["Lips<br/><small>Matte/gloss/velvet finishes<br/>Cosplay tint wash & 0.45 highlight lift</small>"]:::enhancement
        blush["Blush Wash<br/><small>Rosy cheeks & nose tip<br/>Under-eye eyeshadow blend</small>"]:::enhancement
        hair["Hair Shine<br/><small>Exposure, midtone, highlight lifts<br/>Local specular/contrast boosts</small>"]:::enhancement
        dodge_burn["Dodge & Burn<br/><small>Micro sculpting</small>"]:::enhancement
    end

    subgraph ROI_Merge["Portrait ROI Merger"]
        blend_back["ROI Blend Back<br/><small>Pasting processed ROI and masks back<br/>using feathered skin+hair mask</small>"]:::detection
    end

    subgraph Stage13_17["Stages 13-17: Global Tonal & Grading Finishing (Full Canvas)"]
        tonal["Global adjustments<br/><small>Contrast, brightness (gamma)<br/>Tonal curves (highlights, shadows, whites, blacks)</small>"]:::finishing
        color_transfer["Subject-Aware Color Transfer<br/><small>Reference-based Skin, Hair, Background matching<br/>Reinhard ratio limits [0.3, 3.0]</small>"]:::finishing
        grading["Color Grading<br/><small>Split-toning, HSL, presets/stacking<br/>White costume pearl/lavender lift</small>"]:::finishing
        sharpening["Selective final sharpening<br/><small>Unsharp masking over face/hair/eyebrow edges</small>"]:::finishing
        impact["Global high-impact finish<br/><small>Luminance curves, saturation, clarity, glow</small>"]:::finishing
    end

    output["Output Image (BGR)"]:::inputOutput

    %% Connections
    input --> detector
    input --> person_mask
    detector --> slimming
    slimming --> roi_extract
    person_mask --> roi_extract
    roi_extract --> coord_shift
    coord_shift --> bisenet
    bisenet --> faceregions
    
    faceregions --> freqsep
    freqsep --> smoothing
    
    smoothing --> skin
    skin --> blemish
    blemish --> undereye
    undereye --> neck
    neck --> eyes_teeth
    eyes_teeth --> lips
    lips --> blush
    blush --> hair
    hair --> dodge_burn
    
    dodge_burn --> blend_back
    blend_back --> tonal
    tonal --> color_transfer
    color_transfer --> grading
    grading --> sharpening
    sharpening --> impact
    impact --> output
```

---

## 3. Core Modules & Stage Breakdown

### 3.1. Face Detection & Landmarking (`retouch/detection.py`)
*   **RetinaFace (Primary)**: Detects face bounding boxes.
    *   *Keras 3 Runtime Fix*: Sets `os.environ["TF_USE_LEGACY_KERAS"] = "1"` at initialization. This avoids silent execution crashes due to symbolic tensor serialization errors under Keras 3.x / TensorFlow 2.16+.
*   **Crop-based Fitting**: Crops face regions with 30% padding and runs `FaceLandmarker` on the crop. Landmarks are remapped back to full-image coordinates. This makes landmark fitting highly robust on side profiles and distant subjects.
*   **MediaPipe Landmarker (Fallback)**: If RetinaFace fails or is unavailable, the system runs the landmarker on the full image.
*   **Explicit CPU Delegation**: To guarantee cross-platform execution stability and avoid silent execution hangs/crashes, the underlying MediaPipe Landmarker and Image Segmenter pipelines are initialized with explicit CPU delegate configuration (`base.Delegate.CPU`).

### 3.2. Face Reshaping / Slimming (`retouch/geometry.py`)
*   **FaceReshaper**: Applies photographer-grade local translation warping (liquid warping) to jawline landmarks 234 & 454 (inward shift of 2%–5%), cheeks 117 & 346 (inward shift of 1%–3%), and chin 152 (upward shift of 1%–2%).
*   **Parallel Execution Grid**: Accumulates coordinate displacements for all faces first, executing a single `cv2.remap` for zero-overhead performance.

### 3.3. Makeup Engine (`retouch/makeup.py` & `retouch/lips.py`)
*   **Blush Engine (`retouch/makeup.py`)**: 
    *   Generates heavily feathered radial masks around cheek landmarks 117 & 346, applying a rosy flush by nudging the LAB `a` channel positively.
    *   **Nose tip blush**: Supports a cute circular nose tip wash centered at landmark 4 (nose tip).
    *   **Under-eye blush**: Integrates an anime-style eyeshadow blend utilizing feathered left and right under-eye masks.
    *   **Color contamination exclusion**: Automatically subtracts the lip mask from the blush mask to prevent blush shifts from modifying lip colors.
    *   **Vibrant scaling**: Increased maximum shift factor from `12.0` to `16.0` for vivid, clean cosplay looks.
*   **Lip Tint & Specular Upgrades (`retouch/lips.py`)**:
    *   Supports `matte` (pure color tint preserving lip textures), `gloss` (adds specular gloss reflection highlights), and `velvet` (bilateral smoothing on lip textures and reduced opacity texture overlay).
    *   **Cosplay tint wash**: Applies a strong, anime-accurate tint wash at up to 85% opacity (instead of the standard 25% max) for the cosplay recipe.
    *   **Highlight Pop**: Increased highlight lift factor from 0.25 to 0.45, making specular highlights on lips pop prominently.

### 3.4. Face Parsing & Region Masking (`retouch/parsing.py`)
*   **BiSeNet ResNet18 ONNX**: Performs pixel-precise semantic segmentation of face parts. Output categories (skin, eyebrows, eyes, mouth interior, lips, neck, hair) are converted to float32 masks.
*   **Adaptive Feathering**: Feathers mask edges dynamically based on the **Inter-Eye Distance (IED)** to guarantee seamless blending during skin adjustments.
*   **Mask Cleaning Post-Feathering**: Cleans the skin mask after feathering by subtracting the eyebrows, eyes, lips, and mouth interior masks. This prevents skin whitening/equalization filters from bleeding into facial features.
*   **Landmark Fallback**: If ONNX inference is bypassed, it generates polygon-based region masks using specific MediaPipe landmarks.

### 3.5. Frequency Separation & Smoothing (`retouch/frequency.py` & `retouch/skin.py`)
*   **3-Level Separation**: Splits the image into coarse (color/tonal flow), medium (minor skin structures), and fine (pore details/hair strands) frequency bands using Gaussian blur radius scaled to the face width.
*   **Bilateral Filtering & Mid-Frequency Reduction**: Smooths low/medium bands to level out skin blotchiness while conserving high-frequency details. Performs bilateral filtering directly on the `float32` representation to prevent precision loss and preserve micro-contrast. Restricts the hybrid Gaussian blend factor to `smooth_strength * 0.25` (instead of `1.25`) to ensure bilateral filtering remains dominant and avoids washing out fine textures. Introduces the `mid_reduction` parameter to target minor skin structures and blemish anomalies without creating a plastic look.
*   **Independent Nose Smoothing**: Supports a dedicated `nose_smooth` override to allow separate control over the nose bridge texture smoothing versus the rest of the face.

### 3.6. Component-Level Enhancers
*   **Blemish Removal (`retouch/blemish.py`)**: Identifies high-frequency blemishes on the skin mask and applies fast marching inpainting.
*   **Eyes (`retouch/eyes.py`)**: Whitens the sclera (using LAB luminance boosts) and sharpens/saturates the iris. Boosts catchlights and reflections by up to 25%.
*   **Teeth (`retouch/teeth.py`)**: Segments the mouth interior and whitens/desaturates yellow-hued pixels in the LAB/HSV color space.
*   **Hair & Outfit (`retouch/hair.py`)**: 
    *   Segments hair and outfit boundaries using the selfie segmenter.
    *   **Luminance lifts**: Applies region-wide exposure (+8%), midtone (+12%), and highlights (+10% on pixels > 128) lifts to the LAB L-channel.
    *   **Specular and contrast boosts**: Follows with local specular highlight Morped dilation and local contrast boosts (via unsharp masking) to achieve silky hair strand separation.
*   **Under-Eye & Sculpting (`retouch/undereye.py` & `retouch/skin.py`)**: Lightens dark circles and applies subtle Dodge & Burn contours to the nose bridge, forehead center, cheeks, and jawline.
*   **Skin Foundation & Equalization (`retouch/skin.py`)**:
    *   **Adaptive Rosy Foundation (`whiten`)**: Performs soft-clipping skin whitening and rosy/porcelain color shifts, utilizing a clean pre-modification reference copy (`lab_original`) for target medians. Implements shadow protection during foundation/whitening shifts (applied only to regions where $L > 80$) to prevent bruised or purple shadows in darker areas. Supports negative strength for skin darkening (tanning/moody look) with a shadow-preserving decay.
    *   **Skin Tone Equalization (`equalize`)**: Equalizes skin tone using local average color harmonization and CLAHE luminance leveling. Features highlight protection (`_get_highlight_protection`) to avoid clipping highlight areas. Harmonizes color using strength-aware pull values (`0.20 * s * skin_mask`) and blends back onto the skin mask.
    *   **Face-to-Neck Harmonization (`harmonize_neck`)**: Matches neck/chest skin tone to the face to prevent white face / dark neck discrepancies. Features crash protection against empty face landmarks, and applies an adaptive Gaussian blur kernel to the neck mask based on face size. Supports bi-directional neck corrections (neck lighter or darker than the face).

### 3.7. Color Grading & Global Finishing (`retouch/grading.py` & `retouch/engine.py`)
*   **Luminance Curves**: S-curves applied to the LAB L-channel to shape contrast.
*   **Parametric Tonal Adjustments**: Adds direct overrides for highlights, shadows, whites, and blacks via parametric LUT curve adjustments.
*   **Global Brightness**: Leverages gamma-curve lookup tables to correct exposure.
*   **Reference-Based Color Transfer**: Matches the color tone and palette of an uploaded reference image using local distribution adjustments.
*   **Split Toning & Presets**: Colorizes shadows and highlights independently. Introduces 3-way split toning (shadows, midtones, highlights) in LAB space, and supports preset weights stacking (`grade_stack`).
*   **Smart Glow (Orton & Bloom)**: 
    *   *Smart Bloom*: Screens a blurred highlights layer ($L > 225$) back onto itself, restricted to a spatial mask to protect hair and eyes.
    *   *Orton Glow*: Blends a soft Pegtop-blended layer for dreamy fantasy aesthetics.
*   **Highlight Costume Lift**: Lifts high-luminance ($L > 170$) low-saturation white colors (excluding skin, hair, and lips) with soft feathering in cosplay-oriented presets (e.g. `pink_dream`, `meitu_clone`) to recover costume details.
*   **Vignetting, Clarity & Lens Effects**: Micro-contrast (clarity) via guided filtering, radial vignetting, film grain, LUT emulations, halation, and radial chromatic aberration.
*   **Selective Final Sharpening**: Photoshop-style selective unsharp masking over a soft mask (targeting eyes, eyebrows, and hair edges) with custom radius, amount, and threshold settings to finalize high-frequency details.
*   **Global High-Impact Finish**: A dedicated finishing pass (`add_impact_finish`) that uses luminance curves, saturation boosts, micro-contrast clarity, and pink-tinted glow to add global punch.

### 3.8. Portrait Style Cloning & Subject-Aware Matching (`retouch/style.py`)
*   **`StyleProfile`**: A JSON-serializable dataclass representing the extracted style parameters (global brightness delta, global contrast delta, global saturation delta, skin L/a/b color deltas, skin smoothness, mid-frequency reduction, and texture opacity). Includes `save()` and `load()` helpers.
*   **`StyleAnalyzer`**: Extracts a style profile from an aligned Original vs. Edited image pair. Features:
    *   *Percentile-based Contrast*: Uses the `p95 - p5` LAB L range to isolate contrast from exposure shifts.
    *   *Multi-Face Learning*: Accumulates skin masks across multiple faces using a bounding-box-area-weighted average face width to run frequency separation.
    *   *Luminance Grayscale Projection*: Projects multi-channel frequency layers using Rec.601 coefficients to avoid shape mismatch indexing bugs.
*   **`StyleApplier`**: Automatically maps a `StyleProfile` to native `RetouchEngine` parameters (contrast, brightness, smoothing, and whitening tone settings).
*   **`subject_aware_transfer`**: Performs independent, masked Reinhard color matching in the LAB color space for the **Skin**, **Hair**, and **Background** regions between the target image and reference image. Clamps the standard-deviation transfer ratio to `[0.3, 3.0]` to prevent color explosions in uniform regions.

### 3.9. Interactive GUI Dashboard (`gui.py`)
*   **Gradio Web GUI**: Provides a modern, browser-based user interface to interactively process images.
*   **Features**:
    *   Interactive dropdown for selecting pre-configured recipes (which automatically populate sliders).
    *   Side-by-side comparison mode showing original vs retouched outputs.
    *   Manual overrides for all local parameters (smoothing, nose smoothing, whitening, lips/eyes enhancements, Dodge & Burn).
    *   Accordion folders for detailed skin tone adjustments, tone curves, and lens effects.
    *   Reference Image uploader for live color transfer.
    *   **Batch Processing Tab**: Folder ingestion with descriptive grouping, progress bar, contact sheet generation, and ZIP packaging.
    *   **Style Learning Tab**: Dataset-level style extraction from original/edited pairs, saved to `styles/` directory.
    *   **Virtual Studio Relighting Tab**: 3D face relighting controls (azimuth, elevation, strength) using Blinn-Phong shading.
    *   Export resolution overrides (Original, 4K, 2K, 1080p, 720p) and format encoders (JPEG with quality slider, PNG, WebP).
    *   Fast Preview mode running downscaled inference for low latency interaction.

### 3.10. Recipe System (`retouch/recipes.py`)
*   **Structured Presets**: Configures high-level presets (e.g., `natural`, `cosplay`, `scifi_cosplay`, `cyber_doll`, `fuji_porcelain`, etc.) by mapping component parameters to scaling factors.
*   **Recipe Inheritance (`extends`)**: Allows a recipe to inherit from a base recipe (using the `extends` keyword), resolving deep overrides recursively through a deep merging utility (`_deep_merge`).
*   **Specialized Behavior**: Defines sets of recipes that automatically trigger specific engine logic (e.g., `_NOSE_BLUSH_RECIPES` for nose tip blush, `_SLIMMING_RECIPES` for liquid jaw/chin reshaping, and `_WHITE_COSTUME_RECIPES` for white costume highlight recovery).

### 3.11. Processing Pipeline Context & Result Types (`retouch/engine.py`)
*   **`ProcessingContext`**: A typed dataclass that encapsulates all parameters for a processing run, replacing raw dictionaries to provide compile-time safety and self-documenting parameter lists.
*   **`ProcessingResult`**: An ndarray-derived container subclass that acts directly as a standard uint8 BGR image for compatibility with OpenCV/PIL while embedding rich processing metadata:
    *   `skin_mask`: Cumulative normalized skin mask.
    *   `skin_hair_mask`: Combined skin, hair, and neck mask.
    *   `lips_mask`: Lip region mask.
    *   `sharpen_mask`: Soft mask used for final selective unsharp masking.
    *   `face_count`: Number of faces processed.
    *   `params`: The fully-resolved `ProcessingContext` instance.
    *   `timings`: Execution times for each pipeline stage (detection, reshape, per-face, global, grading, finish, total).

### 3.12. Common Mathematical and Mask Utilities (`retouch/utils.py`)
Provides reusable image processing, coordinate geometry, and mask operations:
*   **`feather_mask`**: Feathers binary masks dynamically using Gaussian blur to eliminate hard edges.
*   **`blend_masked`**: Composites a processed BGR image onto the original using a normalized float32 mask.
*   **`correct_exposure`**: Adjusts global luminance distributions using histogram shifts.
*   **`vibrance`**: Selectively adjusts BGR color saturation inside a mask while avoiding over-saturation of skin tones.
*   **`inter_eye_distance`**: Computes distance between pupils to scale filters relative to face size.

### 3.13. Image Input/Output & RAW Format Handler (`retouch/io.py`)
Provides a decoupled, reusable I/O boundary that handles file read/write, format conversions, and metadata preservation:
*   **RAW Image Support**: Integrates `rawpy` to decode professional camera RAW formats (e.g., `.raf`, `.cr2`, `.nef`, `.arw`, `.dng`) directly into BGR arrays.
*   **EXIF Metadata Restoration**: Automatically transposes image orientations via `PIL.ImageOps.exif_transpose` during loading, and copies EXIF headers from original to processed outputs during saving using `copy_exif` (while normalizing the orientation tag).
*   **Stitched Comparison Generator**: Generates high-resolution side-by-side comparison images using a light-gray vertical separator.
*   **Adaptive Scale Limiting**: Rescales large canvases to match processing thresholds while tracking scale factor ratios.
*   **Shared Helpers**: Also exports `resize_for_processing`, `output_format`, `encode_write_params`, and the `IMAGE_EXTENSIONS` / `RAW_EXTENSIONS` sets used by CLI, GUI, and batch tooling.

### 3.14. Virtual Studio Relighting (`retouch/relight.py`)
Implements directional 3D shading based on a FaceMesh depth map derived from MediaPipe landmarks:
*   **Delaunay Triangulation Depth Map**: Builds a depth map (`Z_pixels`) by averaging MediaPipe landmark `z` coordinates across Delaunay triangles, scaled by `face_width` for aspect-ratio correctness.
*   **Surface Normal Estimation**: Computes per-pixel surface normals via Sobel gradients on the blurred depth map.
*   **Blinn-Phong Lighting**: Applies diffuse (ambient-clamped `I_diffuse`) and specular (`I_specular^alpha`) components from configurable `azimuth` and `elevation` light angles.
*   **Yaw Guard**: Attenuates the effect on extreme profiles (`ratio > 1.7`) to prevent unnatural lighting on side profiles.
*   **Highlight Protection**: Decays specular contribution from `L > 220` to `L = 250` to avoid blowing out bright skin areas.
*   **Skin-Masked Compositing**: Blends the relit result onto the original using the soft skin mask.

### 3.15. Batch Processor (`retouch/batch_processor.py`)
Folder-based batch ingestion with descriptive grouping, persistent caching, and contact sheet generation:
*   **`BatchProcessorCache`**: Persistent JSON cache keyed by SHA-256 of the input directory path, stored under `~/.cache/retouch/`. Caches per-file face count, face width, HSV stats, and modification timestamps to avoid re-analysis on repeat runs.
*   **`classify_image`**: Rule-based grouping into 8 categories (Portrait/Scenic × Bright/Dark × Colorful/Muted) using HSV statistics and face counts.
*   **`analyze_and_group`**: Scans all images in a directory using the engine's detector, computes HSV stats (optionally masked to the subject via person segmentation), and groups them by the classification scheme.
*   **`generate_contact_sheet`**: Produces a tiled grid of processed thumbnails with PIL-rendered filename labels and empty-cell placeholders.
*   **`BatchProcessor.process_folder`**: Top-level ingestion → analysis/grouping → style processing → contact sheet → optional ZIP packaging, with a `progress_callback` for GUI integration.

### 3.16. Style Library & Dataset Learning (`retouch/style_library.py`)
Dynamic preset management and dataset-wide style extraction:
*   **`save_style_profile`**: Serializes a `StyleProfile` (or raw dict) with metadata (name, author, version, tags, timestamp) to a JSON file under `styles/`. Auto-increments version on duplicate names to protect history.
*   **`load_style_profile`** / **`list_styles`**: Load individual or enumerate all saved style profiles with metadata.
*   **`normalize_stem`**: Strips Lightroom-style suffixes (`_edit`, `_edited`, `_retouched`, `_v1–_v3`, `_crop`, `_copy`) to match original/edited pairs by stem.
*   **`learn_dataset_style`**: Walks original and edited directories, matches pairs via normalized stems, runs `StyleAnalyzer.extract` on each, filters zero-delta profiles (failed face detection), and averages parameters using a trimmed mean (top/bottom 10% removed when N ≥ 5) to produce a single averaged style profile.

### 3.17. Command Line Interface (`cli.py`)
Provides batch processing capabilities and pipeline customization via command line options:
*   **Batch Directory Recursion**: Recursively resolves inputs (`find_images`) to batch-process folders of target images.
*   **Multiprocessing Engine**: Leverages Python's `multiprocessing` to process multiple images in parallel across CPU cores using pool-based task scheduling.
*   **Parameter Mapping**: Builds parameter configurations (`build_params`) from arguments and applies recipe defaults with runtime overrides.

---

## 4. Model Configurations & Execution Providers

ONNX models are located in `models/` and initialized with hardware acceleration options:
*   **resnet18.onnx** (BiSeNet): Prefers `CoreMLExecutionProvider` on Apple Silicon macOS, falling back automatically to `CPUExecutionProvider` on other architectures.
*   **face_landmarker.task**: MediaPipe task binary.
*   **selfie_segmenter.tflite**: TensorFlow Lite segmenter.
*   *Note*: Underlying MediaPipe elements delegate execution to `CPU` specifically to ensure cross-platform runtime reliability.

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

---

## 6. Performance & Parallel Execution Framework

To handle large-scale images (up to 24 MP and higher) in near real-time, the engine implements two main performance optimization pathways:

### 6.1. Padded Portrait ROI Extraction (Local Processing)
Instead of executing CPU-intensive operations (such as high-dimensional semantic parsing, 3-level frequency separation, and bilateral/guided filtering) on the full resolution canvas:
1. **BBox Padding**: An expanded region of interest (ROI) enclosing the face, hair, neck, and upper chest is calculated dynamically from the face bounding box (padding top by 80%, bottom by 180%, and sides by 60% of face height/width).
2. **Coordinate Relocalization**: Bounding box coordinates and landmark coordinates are mapped relative to the cropped ROI.
3. **Execution**: Semantic segmentation, Bilateral frequency separation, and component enhancement logic operate exclusively on the crop.
4. **Feathered Blending**: The processed crop is blended back into the original full-resolution canvas using soft skin+hair masks.

This optimization cuts peak memory from **7.5 GB to 1.84 GB** (a 4x reduction) and total runtime from **15.3s to 3.09s** (a 5x speedup) on 24 MP images.

### 6.2. Multi-Face ThreadPool Parallelization
For images containing multiple faces:
1. **Parallel Execution**: The per-face processing stage (`_stage_per_face`) distributes the execution of `_process_one_face` across threads using Python's `ThreadPoolExecutor`.
2. **Worker Pool Cap**: Workers are capped at `min(len(faces), 4)` to avoid CPU thrashing and memory overhead.
3. **Thread-Safe Slicing**: Each thread operates on its own ROI crop, and writes results into a list of independent `_FaceResult` structures, which are merged back serially in `_composite_faces`.

---

## 7. Changelog

| PR/Commit | Module | Fix |
|---|---|---|
| BUGFIX-1 | `engine.py` | `s_mask` multi-face scope leak — accumulated skin mask now correctly passed to split-tone mask |
| BUGFIX-2 | `engine.py` | `color_ref` double-write — removed erroneous early `None` assignment shadowing the kwarg |
| BUGFIX-3 | `engine.py` | `r_dark_circles` reused `"whites"` key instead of dedicated `"dark_circles"` |
| BUGFIX-4 | `engine.py` | LUT name collision — internal ndarray renamed to `_brightness_lut` |
| BUGFIX-5 | `frequency.py` | `dodge_burn` mask dtype — force `float32` accumulator to prevent silent uint8 truncation |
| BUGFIX-6 | `skin.py` | `equalize` blend strength — final `blend_masked` now uses `skin_mask * s` for consistent strength scaling with AB harmonization |
| BUGFIX-7 | `grading.py` | `_add_halation` accepts `float`/`int` as direct intensity (not just dict) |
| BUGFIX-8 | `frequency.py` | `np.random.RandomState` replaced with `np.random.default_rng` (modern PCG64) |
| BUGFIX-9 | `frequency.py` | Hash seed sign safety — added `abs()` guard before `& 0xFFFFFFFF` |
| BUGFIX-10 | `skin.py` | `whiten` blend masking — added missing `blend_masked` with skin mask |
| BUGFIX-11 | `skin.py` | `equalize` AB harmonization — pull scaled by `s` and restricted to skin mask |
| BUGFIX-12 | `skin.py` | `dodge_burn` — added `blend_masked` compositing with skin mask |
| BUGFIX-13 | `skin.py` | `specular_bloom` porcelain tone — added porcelain LAB shift branch |
| BUGFIX-14 | `skin.py` | `harmonize_neck` — bi-directional adjustment (brightens neck when darker, darkens when lighter) |
| BUGFIX-15 | `skin.py` | `specular_bloom` — clamp highlight mask to skin boundary to prevent bleed into hair/eyes |
