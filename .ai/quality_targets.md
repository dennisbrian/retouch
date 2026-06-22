# Quality Targets: Aesthetic Guardrails & Tuning Rules

To maintain high visual quality and avoid typical automated-editing artifacts (e.g. waxy skin, purple shadows, or unrealistic lighting), the pipeline adheres to the following quality rules:

## 1. Skin Retouching Aesthetics
- **Avoiding the "Waxy" Look**:
  - Restrict the hybrid Gaussian blur blend factor to `smooth_strength * 0.25` in [frequency.py](file:///Applications/htdocs/retouch/retouch/frequency.py). Bilateral filtering on float32 representation must remain dominant to preserve micro-contrast.
  - Use `mid_reduction` to target isolated blemish anomalies without flattening fine skin textures.
  - Add synthetic noise/pore textures via `pore_synthesis` when high-strength smoothing is requested.
- **Shadow Protection in Foundation Whitening**:
  - In [skin.py](file:///Applications/htdocs/retouch/retouch/skin.py#L169), foundation whitening shifts (`whiten` parameter) must be restricted to pixels with a LAB luminance channel $L > 80$.
  - Darker shadows must decay to zero shift to prevent introducing bruised, purple, or desaturated dark areas on skin.
- **Highlight Protection in Equalization**:
  - Local color/luminance equalization must apply highlight-protection thresholds to prevent blowing out natural skin highlights (e.g., forehead or cheek catchlights).

## 2. Face Detection & Landmarking Accuracy
- **Symmetrical Cropping**:
  - When cropping face ROIs for per-face stages, apply symmetrical padding relative to the bounding box to keep the face centered and prevent cropping out hair or ears.
- **Detection Fallback Target**:
  - Target a > 95% detection rate. If RetinaFace crop landmarking fails (due to angles or profile occlusion), automatically trigger the full-image MediaPipe detection pass as a fallback.
- **Face-to-Neck Harmonization**:
  - Blending masks for the neck must use a Gaussian blur radius dynamically proportional to the inter-eye distance (IED) to ensure a seamless color transition with the chest.

## 3. Lighting & Specular Finishes
- **Relighting Yaw Guard**:
  - In [relight.py](file:///Applications/htdocs/retouch/retouch/relight.py), directional 3D relighting must decay to zero on extreme face profiles (profile ratio > 1.7) to avoid projecting physically impossible shadows.
  - Decay specular shading contribution from $L = 220$ to $L = 250$ to avoid clipping bright spots.
- **Lip Finishes**:
  - Matte finish must apply color tints while conserving original texture; gloss finish must lift catchlights by up to 25% to simulate specular shine.
