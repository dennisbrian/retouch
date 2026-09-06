# Plan — Face Retouch Algorithm Research Execution

**Date:** 2026-09-05  
**Status:** `active execution` — FA-01 core protection wiring and smoothing validation completed; ontology specification, SegFace challenger comparison, and FA-02 through FA-07 phased tranches scheduled.  
**Research basis:** [Face retouch algorithm research](RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md)  
**Parent audit:** [FA-01 protection-consistency and abstention-observability audit](RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md)  
**Program basis:** [Face Retouch Quality-Ceiling Implementation Plan](PLAN_FACE_RETOUCH_QUALITY_CEILING_IMPLEMENTATION_2026_09_01.md)

---

## 1. Executive Summary & Foundational Principles

This document governs the engineering execution of the seven research experiments defined in [Face retouch algorithm research](RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md) (FA-01 through FA-07). It bridges algorithmic research, empirical validation, and codebase implementation within the **Photographic Truth Core** architecture.

### Core Architectural Rules
1. **Source-Pixel Render Authority:** Source pixels are the sole ground truth. No generative hallucination or unconstrained diffusion inpainting is permitted in the core pipeline.
2. **Selective Localization Over Stacking:** Improve *where* an operation acts before adding general smoothing or filtering models.
3. **Decoupled 2-Factor Validation:** Never evaluate detection and repair as a single compound metric. Evaluate localization and repair independently before testing joint interactions.
4. **Strict Abstention Observability:** Any operation back-off, scale damping, or regional omission must log its exact quantitative reason rather than silently dropping computed signals.
5. **Deterministic Legacy Equivalence:** Default recipes without explicit opt-ins (`mark_policy=None`, legacy smoothing) must remain bit-identical or byte-identical to established golden test hashes.

---

## 2. Status Ledger & Verified Progress to Date

The program executes in strict dependency order: Support & Protection (FA-01) → Texture Restoration (FA-02) → Blemish Matrix (FA-03/04) → Reference/Material Decomposition (FA-05/06) → Bounded Learned Assistance (FA-07).

| Tranche | Identifier / Commit | Scope & Verified Deliverables | Status |
|---|---|---|---|
| **FA-01.1** | [`6b1e72d`](https://github.com/example/commit/6b1e72d) | **Boundary Confidence Exposure:** Added boundary-ring parse confidence in `parsing.py`. Observation-only contract; no render changes. | **LANDED** |
| **FA-01.2a** | [`f3b1008`](https://github.com/example/commit/f3b1008) | **Mark Policy in Blemish:** Wired `mark_policy` preserve mask into `blemish.remove` in `perf_optimizations.py`. | **LANDED** |
| **FA-01.2b** | [`154d854`](https://github.com/example/commit/154d854) | **Mark Policy in Evening Ops:** Wired `skin_n_marks_protected` into 9 evening skin ops (`blotch`, `redness`, `yellowness`, `pores`, `oil`, `roughness`, `melanin`, `hydration`, `firmness`). | **LANDED** |
| **FA-01 Audit** | [`5bac5ac`](https://github.com/example/commit/5bac5ac) | **FA-01 Protection & Abstention Audit:** Established base smoothing had zero mark protection (byte-identical render verified) and discovered discarded abstention reasons in `RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md`. | **LANDED** |
| **FA-01.2c** | [`8a2869e`](https://github.com/example/commit/8a2869e) | **Abstention Logging:** Added per-face debug logging for `utils.yaw_gate_factor` and `eye_artifact_safety.assess_eye_artifact_scales`. | **LANDED** |
| **FA-01.3 Exp** | [`c492291`](https://github.com/example/commit/c492291) | **Smoothing Mark Protection Experiment:** Compared 3 mechanisms in `scripts/qa/smoothing_mark_protection_experiment.py`. Validated `blend_alpha` (recovers 74–78% contrast with <0.35 halo) and rejected `filter_input` (~10x larger halo). | **LANDED** |
| **FA-01.3 Impl** | Working tree / §1.1b | **Guided Smoothing Mark Protection:** Implemented `mark_protect` parameter in `FrequencySeparator.combine()` with per-blob distance-transform feathering (`MARK_PROTECT_FEATHER_FACTOR = 0.6`). 16 dedicated unit tests added in `tests/test_frequency_mark_protection.py`. | **SHIPPED** |
| **FA-01.4** | Specification | **Formal Material Ontology & Evidence Fields:** Multi-class label definitions, boundary uncertainty margins, and explicit abstention field contracts. | **READY** |
| **FA-01.5** | Diagnostic | **SegFace vs BiSeNet Benchmark:** Evaluate boundary-ring error and accessory/hair containment on CPU/MPS runtime. | **READY** |
| **FA-02** | Phase 2 | **Selective Skin Texture Restoration:** Defect-excluded texture residual, band-selective recovery, noise vs pore ablation. | **QUEUED** |
| **FA-03 / 04** | Phase 3 | **Blemish Detection × Repair 2-Factor Study:** Candidate generators (dark vs chroma vs learned) × repair methods (Telea vs PatchMatch vs learned). | **QUEUED** |
| **FA-05 / 06** | Phase 4 | **Anatomical Reference & Material Decomposition:** Under-eye reference ring & makeup exclusion + CGFR Gaussian spot fitting vs chromophore/intrinsic tone. | **QUEUED** |
| **FA-07** | Phase 5 | **Bounded Learned Assistance:** Bilateral space (InstantRetouch) & diffusion/LoRA studio boundary gates. | **QUEUED** |

---

## 3. Phased Research Execution Queue

```text
Phase 1 (FA-01): Support, Protection & Boundary Ownership (85% Completed)
  ├── 1.1–1.3: Core mark wiring, smoothing protection, abstention logging (DONE)
  ├── 1.4: Material ontology & evidence-field contract specification (ACTIVE)
  └── 1.5: SegFace vs BiSeNet boundary benchmark (NEXT)
Phase 2 (FA-02): Selective Skin Detail & Texture Restoration
  ├── 2.1: Defect-excluded texture residual computation
  ├── 2.2: Band-selective high-frequency recovery (pore vs noise isolation)
  └── 2.3: Multi-arm ablation study (disabled vs global vs band-selective)
Phase 3 (FA-03 & FA-04): Blemish Localization × Repair 2-Factor Matrix
  ├── 3.1: Independent candidate detectors (luminance anomaly + chroma contrast)
  ├── 3.2: Local repair engines (Telea vs restricted PatchMatch vs learned)
  └── 3.3: 3×3 factor-isolated matrix evaluation on labeled corpus
Phase 4 (FA-05 & FA-06): Anatomical Reference & Material Decomposition
  ├── 4.1: Under-eye tear-trough reference ring with lash/makeup exclusions (FA-05)
  └── 4.2: CGFR-style Gaussian parameter fitting vs chromophore tone evening (FA-06)
Phase 5 (FA-07): Bounded Learned Assistance & Generative Retouch Gates
  ├── 5.1: Bilateral-space instruction-guided retouching (InstantRetouch)
  └── 5.2: Generative studio quarantine & strict render-authority constraints
```

---

### Phase 1: FA-01 — Support, Protection & Boundary Ownership

**Objective:** Guarantee that all facial operations respect physical feature boundaries, protect non-skin materials (hair, makeup, accessories, lashes), preserve intentional marks (moles, freckles, beauty marks), and log structured abstention records when confidence is low.

#### Completed Milestones
- **Observation-Only Parse Confidence (`6b1e72d`):** Added `parse_boundary_confidence` across 19 BiSeNet classes in `retouch/parsing.py`.
- **Mark Protection in Evening Ops (`f3b1008`, `154d854`):** Wired `mark_policy` preserve mask into `blemish.remove` and 9 evening skin passes in `retouch/perf_optimizations.py`.
- **Smoothing Mark Protection (`c492291`, §1.1b):** Added `mark_protect` to `FrequencySeparator.combine()` with `_mark_protect_feather_mask()` using `cv2.distanceTransform`. Proved 74–78% contrast retention with near-zero halo (<0.35 delta). Validated scope for `smooth_engine == "guided"` only.
- **Abstention Observability (`8a2869e`):** Added structured logging for `yaw_gate_factor` and `assess_eye_artifact_scales`.

#### Remaining Work Packages

##### Work Package FA-01.4: Formal Material Ontology & Evidence Fields Spec
- **Contract:** Specify 10 explicit material classes:
  1. `skin_clear` (unblemished skin eligible for smoothing/evening)
  2. `skin_blemish` (temporary acne/spots eligible for repair)
  3. `skin_mark_stable` (moles, freckles, beauty marks protected by policy)
  4. `hair` (scalp hair, eyebrows, sideburns, bangs)
  5. `eye_lash_margin` (eyelids, lash lines, tear troughs)
  6. `lip_makeup` (lips, gloss, contour lines, lipstick)
  7. `cosmetic_art` (drawn blush, anime cosplay markings, face paint)
  8. `accessory` (glasses frames, piercings, headbands, jewelry)
  9. `facial_hair` (beard, mustache, stubble)
  10. `occlusion_unknown` (hands, clothing, microphone, props)
- **Evidence Fields Schema:**
  - `hard_mask`: uint8 binary mask from argmax.
  - `boundary_margin`: float32 spatial top-1 minus top-2 posterior margin.
  - `protection_mask`: float32 attenuation mask [0.0, 1.0].
  - `abstention_reason`: Optional structured dictionary (`reason_code`, `metric_value`, `threshold`).
- **File target:** `docs/plans/SPEC_FACIAL_MATERIAL_ONTOLOGY_2026.md`.

##### Work Package FA-01.5: SegFace vs. BiSeNet Boundary-Error Benchmark
- **Challenger:** SegFace (AAAI 2025, Kartik et al.), evaluated against current 512×512 BiSeNet.
- **Hypothesis:** SegFace improves boundary precision on thin structures (eyelashes, eyeliner, hair wisps, glasses frames) without increasing full-frame latency by >3× on Apple Silicon.
- **Protocol:**
  - Evaluate on the 83-photo DSCF corpus and 5 Meitu cosplay pairs.
  - Register boundary alignment against manually labeled ground-truth crops (20 eyes, 20 lips, 20 hairlines).
  - Measure inference time on Mac CPU and MPS backends.
  - **Decision Gate:** If SegFace reduces boundary leakage into lashes/makeup by ≥25% with runtime ≤180ms per face on MPS, adopt SegFace behind an experimental model flag (`--face-parser=segface`). Otherwise retain BiSeNet.

---

### Phase 2: FA-02 — Selective Skin Detail & Texture Restoration

**Objective:** Decouple photographed pore-scale high frequencies from noise, compression artifacts, and repaired blemishes, preventing smoothing from producing a "waxy" finish while preventing texture recovery from undoing prior blemish repairs.

#### Problem Statement (Current Code)
In `retouch/skin.py::restore_micro_texture()`, texture restoration computes a difference signal between pre- and post-processed images and blends it back into selected regions. However:
1. The difference image contains both high-frequency pore texture *and* mid-frequency blemishes/blotches that smoothing just removed.
2. Restoring the unfiltered difference can re-introduce the very defect that was smoothed.
3. Sensor noise and JPEG compression ringing in high-frequency bands are amplified if gain is applied globally.

#### Experimental & Implementation Plan

##### Work Package FA-02.1: Defect-Excluded Texture Donor Mask
- Exclude detected blemish supports (`blemish_mask`), protected mark supports (`mark_preserve_mask`), and under-eye exclusion zones from the texture restoration source.
- Ensure only smooth, healthy skin regions donate high-frequency micro-texture.

##### Work Package FA-02.2: Band-Selective Texture Recovery
- Extract high frequencies using Laplacian or guided-filter band separation (`k_size ~ 3–5px` at portrait scale).
- Isolate the true pore band ($F_{high}$) from the blotch band ($F_{mid}$) and noise floor ($F_{noise}$):
  $$\Delta_{\text{safe}} = \text{clamp}\left(F_{high}(I_{\text{source}}) - F_{high}(I_{\text{smooth}}), -\tau, +\tau\right) \times M_{\text{skin}} \times (1 - M_{\text{defect}})$$
- Incorporate adaptive local noise-floor attenuation: where source luminance variance is dominated by Poisson/sensor noise rather than structured pores, dampen restoration gain.

##### Work Package FA-02.3: 3-Arm Ablation Study
- **Arm A (Baseline):** No texture restoration (`restore_micro_texture=0`).
- **Arm B (Current Shipped):** Legacy full-difference texture restoration.
- **Arm C (Challenger):** Band-selective, defect-excluded micro-texture recovery.
- **Evaluation Criteria:** High-frequency energy in clear skin vs blemish sites; blinded visual review of skin naturalness vs waxy appearance; absence of noise halos.

---

### Phase 3: FA-03 & FA-04 — Blemish Localization × Repair 2-Factor Matrix

**Objective:** Separate spot candidate detection from spot filling/inpainting so algorithm improvements can be unambiguously attributed to localization, repair, or their interaction.

```text
                     Factor B: Repair Algorithm
Factor A: Detector    Telea (OpenCV)   Restricted PatchMatch   Learned / Inpainting
-----------------------------------------------------------------------------------
Current Luma (D)         Cell (1,1)          Cell (1,2)             Cell (1,3)
Fixed Chroma (D or R)    Cell (2,1)          Cell (2,2)             Cell (2,3)
Learned Locator          Cell (3,1)          Cell (3,2)             Cell (3,3)
Ground-Truth Control     Cell (4,1)          Cell (4,2)             Cell (4,3)
```

#### Factor A: Spot Candidate Detection (FA-03)
1. **Fix the Redness Branch Logic Bug:**
   - In `retouch/blemish.py::BlemishRemover._detect()`, current code computes `D OR (R AND D) == D`, making the red-HSV branch completely inert.
   - Implement independent candidate union:
     $$M_{\text{candidate}} = \left( M_{\text{dark}} \cup M_{\text{red\_contrast}} \right) \cap M_{\text{skin\_clear}} \setminus M_{\text{protected\_marks}}$$
2. **Cosplay & Makeup Protection Filter:**
   - Differentiate circular acne/comedones from linear eyeliner, intentional freckles, and drawn cosmetic dots using eccentricity and local ring contrast.
3. **Learned Detector Evaluation:**
   - Evaluate lightweight learned spot detection heads trained on high-resolution portrait crops.

#### Factor B: Spot Repair Engine (FA-04)
1. **Telea Baseline (`cv2.inpaint`):** Fast, effective for micro-spots ($\le 5\text{px}$ diameter). Blurs at larger radii.
2. **Restricted PatchMatch:**
   - Constrain donor search space to same-face, similar-depth, similar-lighting skin patches within a localized search radius ($R \le 150\text{px}$).
   - Prevent donor patches from sampling eyebrows, hair, nostrils, or protected moles.
3. **Bounded Learned Inpainting:**
   - Compare small neural inpainting models (e.g., LaMa or EdgeConnect weights restricted to spot bounding boxes).
   - Enforce hard boundary constraint: pixels outside the defect mask are byte-identical.

#### Execution & Stop Conditions
- Test all 12 cells of the matrix on 50 hand-annotated test crops containing both true blemishes and protected marks.
- **Exit Gate:** Adopt the winning combination that achieves $\ge 85\%$ blemish recall with $0\%$ false removal of protected moles and no visible donor repetition seams.

---

### Phase 4: FA-05 & FA-06 — Anatomical Reference & Material Decomposition

**Objective:** Repair under-eye hollows, puffiness, and skin discoloration while respecting 3D anatomical bone structure, facial lighting form, and intentional cosmetic makeup.

#### FA-05: Under-Eye Reference Ring & Exclusion Calibration
- **Research Basis:** The September 2 dark-circle study (`RESEARCH_DARK_CIRCLE_OP_2026_09_02.md`) redesigned under-eye correction using local reference rings.
- **Open Problems:**
  1. Tear-trough reference sampling can pick up cheek blush or contour makeup, skewing target luminance and creating white patches.
  2. Lower eyelash margins and aegyo-sal (periorbital muscle rolls) must not be flattened as dark circles.
- **Execution Plan:**
  - Calibrate dynamic reference-ring sampling: sample reference skin strictly along the zygomatic arch (cheekbone) where illumination matches but makeup is absent.
  - Implement explicit abstention: if no valid, non-makeup reference skin is found within the sample ring, log `abstention_reason: "undereye_reference_contaminated"` and skip correction.
  - Maintain natural residual shadow: cap maximum lighten factor at $65\%$ of the difference to avoid an unnatural flat flashlight look.

#### FA-06: Material Decomposition & Tone Science
- **Research Basis:** Compare existing multi-pass tone algorithms (`skin.py`, `chromophore_v2.py`, `intrinsic.py`) with physics-based layer modeling (e.g., Shuai et al., ICME 2024 / CGFR).
- **Execution Plan:**
  - Audit the interaction between melanin/hemoglobin spectral separation and frequency-domain tone evening.
  - Test CGFR-style Gaussian parameter fitting on localized hyperpigmentation spots: fit a 2D Gaussian profile to the local pigment density and attenuate only the fitted amplitude while leaving the base skin albedo untouched.
  - Prevent stacking conflicts: ensure tone evening does not run on the same pixels that were already normalized by chromophore or shine reduction.

---

### Phase 5: FA-07 — Bounded Learned Assistance & Generative Retouch Gates

**Objective:** Safely evaluate state-of-the-art deep learning retouching models (InstantRetouch, BeautyGRPO, RetouchFormer) while strictly confining their behavior within the Photographic Truth Core contract.

#### Architectural Quarantine & Bounded Execution
1. **Source Render Authority:** Deep models are **never** permitted to generate end-to-end output pixels directly.
2. **Bilateral Space Decomposition (InstantRetouch CVPR 2026):**
   - The neural model predicts 3D bilateral grid affine slicing coefficients, not raw pixels.
   - The slicing and image reconstruction occur on native-resolution source pixels in local bilateral space, guaranteeing edge preservation and zero high-frequency hallucination.
3. **Strict Bounded Residuals:**
   - If a learned model predicts a residual image $\Delta_{\text{model}}$, the final composite is strictly bounded:
     $$I_{\text{out}} = I_{\text{source}} + \text{clamp}\left(\Delta_{\text{model}}, -\delta_{\max}, +\delta_{\max}\right) \times M_{\text{permitted\_support}}$$
4. **Deterministic Fallback:** If GPU/MPS memory exceeds limits, or if inference latency exceeds $500\text{ms}$, the pipeline silently falls back to the classical deterministic pipeline without crashing.

#### Provenance & License Quarantine Gate
- No model weights or reference repositories may be pulled into production without an explicit intellectual property and dependency review (see Section 6).

---

## 4. Multi-Dimensional Evaluation Contract

Every experiment across FA-01 through FA-07 must measure and report results across eight distinct quality dimensions:

| Dimension | Primary Metric / Instrument | Pass / Fail Criterion | Why a Single Score Fails |
|---|---|---|---|
| **1. Requested Correction** | Spot contrast delta or under-eye shadow attenuation ratio on labeled targets. | $\ge 70\%$ reduction of targeted defect visibility. | Returning source unchanged yields 100% fidelity but 0% task accomplishment. |
| **2. Mark Preservation** | Contrast retention of protected marks (moles, freckles) via `_mark_contrast()`. | $\ge 70\%$ source contrast retained; 0 unwanted mark removals. | A high face-recognition cosine similarity score persists even if a mole disappears. |
| **3. Texture Integrity** | Residual high-frequency band energy ($F_{\text{high}}$) in clear skin; pore observability. | Texture ratio within $[0.85, 1.15]$ of source; zero artificial grain patterns. | Noise, sharpen halos, or compression artifacts falsely register as high frequency. |
| **4. Boundary Ownership** | Mean and max pixel delta in protected non-skin regions (lashes, lips, hair). | Delta exactly 0.0 in protected regions; outer halo $\le 0.35$ luma levels. | Aggregated IoU/Dice scores hide small 2-pixel eyeliner smudges that ruin a portrait. |
| **5. Tone & Form** | 3D shape/luminance gradient curvature; chrominance vector shift in $\text{ICtCp}$. | Preserved natural facial shading contour; $\Delta E_{\text{ITP}} \le 2.0$ outside target spots. | Over-smoothed tone flattens cheekbones and nose bridges into a cartoon mask. |
| **6. Geometry & Likeness** | Dense 478 MediaPipe landmark displacement; unwarped difference inspection. | Landmark drift $\le 0.2\text{px}$; zero unintended geometric warping. | Landmark stability alone does not guarantee skin texture likeness. |
| **7. Calibrated Abstention** | True positive back-off rate on edge cases (extreme yaw, heavy occlusion, tiny faces). | 100% of abstentions emit structured debug log with reason code and metric. | Rejecting every difficult image looks safe on paper but provides zero useful utility. |
| **8. Runtime & Memory** | Cold/warm execution time per mega-pixel; peak RSS memory on Mac Apple Silicon. | Total face-ops pipeline $\le 350\text{ms}$ on 24MP image; peak memory $\le 1.2\text{GB}$. | Server GPU benchmark numbers do not reflect pinned local Mac execution. |

---

## 5. Corpus Protocols & Data Split Governance

### Corpus Composition
Evaluations must draw from the verified 83-photo DSCF portrait corpus and the 5 registered Meitu competitor pairs:
- **Corpus Pool:**
  - 40 frontal studio portraits (clear skin, controlled light).
  - 20 cosplay portraits (heavy cosmetic makeup, wigs, colored contacts, false lashes).
  - 15 outdoor / harsh lighting portraits (high contrast, strong shadows).
  - 8 extreme angle / profile / partially occluded portraits (yaw $>45^\circ$, hand near face).
  - 5 registered Meitu application comparison pairs (`natural` and `convention_clear_v1` recipes).

### Data Split Rules
- **Stratification:** Split subjects by physical identity, not frame index. Near-duplicate frames from the same burst shoot must remain in the same split.
- **Partitioning:**
  - **Dev Set (40%):** Algorithm tuning, parameter sweeping, feature exploration.
  - **Calibration Set (30%):** Confidence thresholding, boundary margin calibration, abstention gating.
  - **Held-Out Test Set (30%):** Final verification, blind owner preference reviews, regression sign-off.
- **Rule:** Never tune algorithm thresholds or feather parameters on the held-out test set.

---

## 6. Model Provenance & License Register

Before any third-party model code or weights are downloaded, benchmarked, or imported, they must be registered in this ledger:

| Model / Library | Paper / Source Reference | Code License | Weight License | Local Execution Feasibility | Deployment Recommendation |
|---|---|---|---|---|---|
| **BiSeNet (CelebAMask-HQ)** | Yu et al., 2018 | MIT | Research / Non-commercial | Current production baseline; ONNX runtime CPU/MPS. | **RETAIN** as default baseline. |
| **SegFace** | Narayan et al., AAAI 2025 | Apache-2.0 | Apache-2.0 / Open | PyTorch / ONNX exportable; evaluates in Phase 1 (FA-01.5). | **EVALUATE** behind `--face-parser=segface`. |
| **InstantRetouch** | Wu et al., CVPR 2026 | Apache-2.0 | Apache-2.0 | Bilateral grid inference; fast local MPS execution feasible. | **EVALUATE** in Phase 5 for bounded assist. |
| **BeautyGRPO** | Yang et al., CVPR 2026 | MIT (code) | Custom / Non-commercial | Requires heavy diffusion backbone; high VRAM footprint. | **QUARANTINE** to Generative Studio track. |
| **RetouchFormer** | Wen et al., AAAI 2024 | Research code | Non-commercial | Transformer attention; high compute cost for 24MP inputs. | **DEFER** until bounded proxy viability tested. |
| **PatchMatch** | Barnes et al., SIGGRAPH 2009 | Research code | Proprietary / Non-commercial | Current in-house implementation is clean-room; do not import research code. | **MAINTAIN** existing clean-room implementation. |

---

## 7. Risk Management, File Boundaries & Rollback Contracts

### File Touch Boundaries
To prevent regressions across the broader Retouch application, future implementation tranches are restricted to specific modules:

- **Phase 1 (FA-01):** `retouch/parsing.py`, `retouch/frequency.py`, `retouch/perf_optimizations.py`, `tests/test_frequency_mark_protection.py`.
- **Phase 2 (FA-02):** `retouch/skin.py`, `retouch/frequency.py`, `tests/test_texture_restoration.py`.
- **Phase 3 (FA-03/04):** `retouch/blemish.py`, `retouch/marks.py`, `tests/test_blemish_matrix.py`.
- **Phase 4 (FA-05/06):** `retouch/undereye.py`, `retouch/skin_chromophore.py`, `retouch/intrinsic.py`.
- **Phase 5 (FA-07):** Dedicated isolated package `retouch/experimental/` — no edits to core engine files.

### Rollback Contract
1. Every new feature must be gated by a configuration flag or parameter defaulting to legacy/disabled behavior (e.g., `mark_protect=None`).
2. Golden hash regression tests (`test_golden_pipeline.py`, `test_golden_pipeline_face.py`) must remain bit-identical on legacy recipe paths.
3. If an implementation tranche fails any multi-dimensional metric (Section 4), the change must be reverted immediately rather than patched forward.
