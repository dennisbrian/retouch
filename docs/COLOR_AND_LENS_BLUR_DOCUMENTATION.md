# Retouch Engine Extensions: Feature Documentation

This document describes the implementation details, mathematical models, and integration of the four major retouch engine improvements completed in this phase.

---

## 1. Per-Face Autoclass Heuristics
A demographic classification heuristic was built to automatically suggest preset recipe overrides (`child`, `senior`, `male`, or `female`) upon face detection.

*   **File Location**: [retouch/face_params.py](file:///Applications/htdocs/retouch/retouch/face_params.py#L122-L212)
*   **Classification Criteria**:
    1.  **Child**: Detected by eye size relative to face size (interpupillary distance `ied` / face width `w_face` > 0.43), vertical proportion of facial landmarks, and aspect ratio.
    2.  **Senior**: Detected by wrinkles and skin texture, calculated via local Laplacian variance on forehead patch crops.
    3.  **Male vs. Female**: Distinguished by lower-face beard shadow coolness/darkness (CIELAB `b*` / `L*` values) and lip redness contrast (CIELAB `a*` values).
*   **Integration**:
    *   Wired to resolve dynamically when `face_params == "auto"` in both proxy and native processing paths.
    *   Pre-populates Gradio UI states in [gui.py](file:///Applications/htdocs/retouch/gui.py#L725-L738) and supports `--face-params auto` in [cli.py](file:///Applications/htdocs/retouch/cli.py#L419-L422).

---

## 2. Lens Blur (Simulated Depth of Field)
Simulates a premium depth-of-field focus blur where the background becomes progressively blurrier depending on its distance from the focus target.

*   **File Location**: [retouch/background.py](file:///Applications/htdocs/retouch/retouch/background.py#L296-L377)
*   **Mathematical Model**:
    *   **Focus Point**: Defaults to the center of the `person_mask` or the center of the image.
    *   **Depth Map Generation**:
        $$\text{depth} = \text{clip}\left(\frac{\text{distance\_to\_focus}}{\text{max\_distance}}, 0.0, 1.0\right) \times 0.4 + \text{clip}\left(\frac{y}{H}, 0.0, 1.0\right) \times 0.6$$
        $$\text{depth} = \text{depth} \times (1.0 - \text{person\_mask})$$
        This combines radial distance with a vertical ground plane gradient to simulate physical distance while keeping the subject in sharp focus.
    *   **Multi-Stage Blending**: Computes three discrete Gaussian blur levels ($\sigma_{max} \times 0.33$, $\sigma_{max} \times 0.66$, $\sigma_{max}$) and interpolates between them across depth map ranges.
*   **Integration**:
    *   Exposed as the `lens_blur` parameter (range 0–100) in the engine parameter registry, CLI, and GUI.

---

## 3. Sclera Masking Granularity
Bridges the competitor mask granularity gap by exposing eye sclera (whites) as explicit, reusable mask regions.

*   **File Location**: [retouch/parsing.py](file:///Applications/htdocs/retouch/retouch/parsing.py#L115) & [retouch/eyes.py](file:///Applications/htdocs/retouch/retouch/eyes.py#L80-L86)
*   **Implementation**:
    *   Added `left_sclera` and `right_sclera` slots to `FaceRegions`.
    *   Populated them inside `FaceParser._add_landmark_subregions` by performing a subtraction of the iris landmarks from the eye masks:
        $$\text{sclera\_mask} = \text{clip}(\text{eye\_mask} - \text{iris\_mask}, 0.0, 1.0)$$
    *   Updated `EyeEnhancer` to consume these precomputed masks directly, reducing redundant operations.

---

## 4. Selective Color & Calibration Primitives
Enables recipe-level and parameter-level control over selective HSL hue bands and color calibration, exposing Fujipreset characteristics (such as Classic Chrome's green-to-teal shift) directly to the engine and presets.

*   **File Location**: [retouch/params.py](file:///Applications/htdocs/retouch/retouch/params.py#L1788-L1853) & [retouch/engine.py](file:///Applications/htdocs/retouch/retouch/engine.py#L2530-L2608)
*   **HSL Bands**: Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta.
*   **Calibration Channels**: Red, Green, Blue.
*   **Implementation**:
    *   Declared 33 ParamSpec objects mapping HSL hue/sat/lum and Calibration hue/sat/lum parameters.
    *   Wired the grading stage in `engine.py` to extract these parameters, build the nested HSL/Calibration adjustment dictionaries, and overlay them onto the active color grading settings dynamically.
