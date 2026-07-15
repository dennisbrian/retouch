# Documentation Audit Report — Cycle 4 (2026-07-15)

This audit analyzes the alignment between the active codebase parameters (in `retouch/params.py` and `retouch/engine.py`) and the system documentation. It identifies missing API parameters, outdated signature references, and provides recommendations for alignment.

---

## 1. Parameter Coverage Deficit

An automated analysis of `retouch/params.py` (which contains **206** registered parameter specs) against `docs/architecture/API.md` shows that **145 parameters are completely missing from the API Reference**. 

### 1.1 Outdated `RetouchEngine.process()` Signature
The signature documented in `docs/architecture/API.md` is frozen at a much earlier version of the engine and completely lacks support for major parameter sets including detailed shape/reshape sliders, body retouching, tone-adaptive specular rendering, and chromophore/intrinsic operators.

### 1.2 Missing Parameters by Domain

#### 1. Specular & Optical Finish
The tone-adaptive specular finish feature (R12) is fully operational but completely undocumented for API developers:
*   `specular_finish` (default: `"matte"`): Specular rendering style (`"matte"`, `"powder"`, `"dewy"`, `"glass_skin"`).
*   `specular_finish_strength` (default: `0.5`): Blend opacity/strength of the selected specular finish.
*   `specular_recolor` (default: `0.0`): Opacity/intensity of specular highlight color-cast neutralization.

#### 2. Intrinsic & Chromophore Operations
The melanin-hemoglobin decomposition (R7/R10) and albedo-shading unmixing (R9) parameters are entirely missing from documentation:
*   `albedo_even`: Albedo layer smoothing/flattening.
*   `makeup_coverage_even`: Foundation coverage map smoothing.
*   `makeup_cake_reduce`: High-frequency makeup caking suppression.
*   `hemoglobin_smooth`: Hemoglobin-specific redness evening.
*   `mole_protect`: Mask protection for moles/beauty marks from color tools.
*   `vein_attenuate`: Blue/deoxygenated vein signature suppression.

#### 3. Body Retouching
All body-specific parameters added in Phase 2 are undocumented in both the API reference and the recipe guide:
*   `body_smooth` / `body_equalize` / `body_whiten` / `body_match_face`
*   `body_relight` / `body_dodge_burn` / `body_shadow_lift`
*   `body_reshape_shoulder_width` / `body_reshape_hip_width` / `body_reshape_arm_length` / `body_reshape_leg_length` / `body_reshape_torso_width`

#### 4. Advanced Hair Retouching
Specialized hair processing parameters:
*   `hair_deglare` / `hair_ring_position` / `hair_ring_tint`
*   `hair_remove_flyaways` (associated with the `SpotHealer` / flyaway solver)

#### 5. Local Detail & Smoothing Engines
*   `smooth_engine`: Which smoothing filter to run (`"guided"`, `"bilateral"`, etc.)
*   `blotch_reduction` / `regional_modulation`
*   `undereye_shadow_strength` / `undereye_darken_removal` / `undereye_puffiness_reduction`
*   `freckle_removal` / `freckle_preserve_mask`

---

## 2. Recipe Guide Gap (`docs/guides/RECIPE_GUIDE.md`)

`docs/guides/RECIPE_GUIDE.md` is the primary document used by artists and developers to design preset JSON files. 

*   **Deficit:** The **Field Reference** table and lists under Section 28 only list ~40 parameters.
*   **Impact:** Preset creators are unaware of advanced features like `specular_finish` or `blotch_reduction` because they do not appear in the guide's dictionary schema lists.
*   **Remediation:** Both the API and Recipe guides need to automatically or manually pull from the `retouch/params.py` registry to ensure zero-drift parameter tables.

---

## 3. Developer Guidance Gaps

### 3.1 Fitzpatrick Tone-Bias Pattern Guidance
The recent audit of `makeup_unmix.py` and the subsequently fixed `specular.py` highlight gate exposed a recurring bug class:
> **The Tone-Bias Pattern:** Hardcoded absolute luminance/reflectance thresholds (`I > 170.0`, `L > 0.85`, etc.) applied to signals whose base levels scale with skin tone (reflectance-dependent).

While this is documented inside research-planning files (`PLAN_P4_MAKEUP_UNMIX.md` §14 and §15), it is **not** surfaced in the general developer guidelines (`CLAUDE.md`) or contributing guidelines (`docs/CONTRIBUTING.md`). 
*   **Risk:** Future developers or agents writing code in `retouch/` might introduce similar absolute-value filters, re-introducing skin-tone performance disparities.

### 3.2 Unwired/Placeholder Features Guidance
Certain features (like depth-aware `lens_blur`) exist in the codebase but have placeholders or unwired controls:
*   `lens_blur` has a hidden Gradio component state and no CLI flag in `params.py`.
*   These are listed as "done" or "backlog" in task files, but are not documented as "unwired/experimental" in user-facing guides, leading to confusion when developers try to use them from presets or CLI overrides.

---

## 4. Priority Recommendations

1.  **Surface Tone-Bias Warning to Developers (High Priority):**
    Append the **Tone-Bias Pattern rule** to [CLAUDE.md](file:///Applications/htdocs/retouch/CLAUDE.md) so that all coding agents and human developers are explicitly warned against hardcoding absolute intensity thresholds.
2.  **Generate Parameter Tables Automatically (Medium Priority):**
    Write a build utility or script that reads `retouch/params.py` and generates/injects the markdown parameter tables directly into [API.md](file:///Applications/htdocs/retouch/docs/architecture/API.md) and [RECIPE_GUIDE.md](file:///Applications/htdocs/retouch/docs/guides/RECIPE_GUIDE.md). This prevents documentation drift permanently.
3.  **Update `RetouchEngine.process()` Signature (Medium Priority):**
    Align the signature block in `API.md` with the actual 100+ argument keyword-only list from `retouch/engine.py`.
