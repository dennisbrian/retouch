# P4 decision: bounded, category-specific makeup appearance editing

Date: 2026-09-06. Original research scope: no production code, parameter, recipe,
or GUI changes. A conservative implementation follow-through was appended on
2026-09-07; the research conclusions and evidence boundaries below remain the
design authority.

Continues [the weekly handoff](../WEEKLY_PROGRESS_2026_09_06.md) and consolidates the code rebaseline and targeted literature already inspected before the interruption. It does not repeat the P4 historical survey or reopen FA-02/FA-03. The separately appended self-blend tone-curve brief is outside this report.

## 1. Decision

Choose **bounded makeup attenuation + category-specific operations + automatic abstention for non-identifiable cases**. Do not retain “recover the skin underneath makeup from one RGB photograph” as P4's product promise.

More precisely:

- **P4 appearance attenuation:** an opt-in, local-reference-conditioned reduction of a confirmed cosmetic's visible color/contrast contribution. Output is an edited photograph, not recovered makeup-free ground truth.
- **P4 coverage/finish repair:** even observed cosmetic coverage or reduce an explicitly identified coating artifact while preserving intended paint color, edges and pattern. This is a different objective from attenuation. Caking is not evidence of an identifiable opacity layer.
- **Protected-category policy:** preserve eye/lash makeup, lip makeup, designed face paint and glitter by default. Unknown category, unknown intent, missing credible reference, opacity/occlusion, clipping or inconsistent estimates => abstain from automatic removal.
- Retain the linear mixture and conditional alpha formula as **experimental/known-layer utilities**, not a claim that an unconstrained joint inverse is solved.

This is a research recommendation, not authorization to rename APIs, change defaults or wire a replacement.

## 2. What the inspected implementation actually does

Code pointers below refer to the inspected working tree. The weekly note is correct that tone-relative fixes did not solve identifiability. The existing [P4 plan](PLAN_P4_MAKEUP_UNMIX.md) also records a later compact-artifact safety mode; older large-damage measurements in its historical sections must not be represented as measurements of today's capped path.

### Fitting and editing

- [`estimate_makeup_alpha`](../../retouch/makeup_unmix.py#L158) uses Oklab lightness/chroma, a chromophore reconstruction residual and log-intensity density. Signals are compared with local Gaussian baselines and per-face median/MAD statistics; their **maximum** forms a soft outlier hint. This is not a posterior probability of makeup and not measured physical opacity.
- The hint undergoes mask exclusion, opening, component selection and erosion. Components occupy roughly 0.8–12% of skin area, with a 20% bounding-box limit. Broad foundation is intentionally outside this automatic mode. Skin area also determines the nominal scale; it is not a measured cosmetic thickness.
- [`unmix_makeup`](../../retouch/makeup_unmix.py#L249) averages high-prior observed pixels into **one RGB makeup color for the entire canvas**. Low-prior skin pixels seed a skin fill; the fallback makeup color is `[220,200,210]` BGR. A Gaussian-smoothed blend initializes S. Neither endpoint is independently observed beneath the cosmetic.
- [`closed_form_alpha`](../../retouch/makeup_unmix.py#L227) projects `I−S` onto `M−S`, then clips to [0,1]. The default three iterations update alpha, re-average high-alpha observations into M, and algebraically invert for S. Despite the docstring's “IRLS,” this loop has no robust residual reweighting or joint, spatially regularized objective. The projection is a closed-form **conditional** least-squares step, not a closed-form joint solve.
- If estimated endpoint separation is below six encoded RGB units, the function returns `S=M=I` with the masked prior: a degenerate identity representation, not recovered layers. Otherwise S inversion limits alpha to .95 and clips S to [0,255], while the returned alpha can exceed .95. Those operations can prevent an exact reconstruction.
- [`even_coverage_alpha`](../../retouch/makeup_unmix.py#L353) Gaussian-smooths alpha within its nonzero support; its `guide` argument is unused. [`cake_reduce`](../../retouch/makeup_unmix.py#L367) reduces high-frequency alpha. Neither operation is simply “lower opacity”: smoothing can increase some pixels while decreasing others.
- [`apply_makeup_unmix`](../../retouch/makeup_unmix.py#L401) recomposes and caps each encoded BGR channel's change to ±5. The cap bounds amplitude, not semantic harm, perceptual color difference or truth. Zero strengths return the input directly. There is no makeup-removal slider or calibrated alpha-decrease operator in this entry point.

### Color/physics and safety limitations

The solve/composite uses encoded BGR values, without radiometric linearization. Feature extraction quantizes float inputs to uint8. The [chromophore helper](../../retouch/chromophore.py#L34) uses fixed, deliberately appearance-oriented pigment directions—its hemoglobin direction is intentionally not a strictly physical extinction model. It takes logs of RGB without inverse transfer-function decoding and reconstructs with a face-wide ambient scalar. Its residual can represent lighting, camera rendering, basis error or cosmetics; it cannot certify “non-skin material.” Red cosmetics may fit a redness direction, while genuine skin under colored light may violate it.

The alpha-fitting exclusions include optional mole, semantic and specular masks. Automatic specular exclusion is compactness/brightness based; a user alpha prior bypasses that automatic step. `user_alpha` initializes the fit but is subsequently recomputed, so it is not generally a fixed calibrated opacity constraint.

The caller constructs eye, brow, lip, mouth and hair exclusions, but catches construction failures by proceeding without that exclusion mask. It does **not** forward `mole_mask` or the already available policy-preservation mask to P4. Also, `cake_reduce` can spread alpha into zero-alpha regions, and the product entry does not reapply the complete external protection mask after recomposition. These are code-inspected contract risks to cover in a future isolated experiment, not claims of newly reproduced real-photo damage. Do not fix them in this research task.

### Consumers and reachability

[`perf_optimizations.py`](../../retouch/perf_optimizations.py#L438) invokes P4 before albedo evening and frequency separation. Downstream stages receive only the recomposed canvas; S, M and alpha are not passed onward as trusted physical layers.

The engine and face-parameter plumbing expose `makeup_coverage_even` and `makeup_cake_reduce`. Their [parameter specifications](../../retouch/params.py#L745) default to zero and define CLI/recipe keys. The GUI has zero-valued `gr.State` entries, not dedicated visible controls at those declarations. QA scripts deliberately enable the options; the inspected cosplay-recipe regression checks them off. The standalone historical spike duplicates an earlier detector/formula and is not the current production entry point.

The already-started focused verification completed: `tests/test_makeup_unmix.py` plus `tests/test_cosplay_portrait_recipe.py`: **41 passed**, with 14 dependency deprecation warnings. This is not full-suite or real-image validation. Tests mainly cover synthetic colors, known-endpoint projection, safety gates and defaults. Their six named skin-tone swatches do not establish six-population fairness, and a loose recomposition test does not measure latent skin accuracy.

## 3. Identifiability: what one RGB observation can and cannot say

For a pixel, `I=(1−a)S+aM` gives three observations for seven scalar unknowns: RGB S, RGB M and a. Spatial constancy or regularization can reduce degrees of freedom, but adds assumptions, not measurements.

For any fixed `0<a<1`, choose a small RGB vector d such that values remain valid:

`S' = S+d`, `M' = M−((1−a)/a)d`.

The observed I is unchanged. Thus even **known opacity** does not identify both unknown colors. At a=1, I contains no information about S. At a=0, S=I but M is unknowable. A perfect forward reconstruction cannot distinguish these alternatives.

Only when S and M are independently constrained does the familiar projection estimate a:

`a = clip(((I−S)·(M−S))/||M−S||², 0, 1)`.

Endpoint similarity makes this unstable. For known endpoints and isotropic observation noise, alpha variance scales with `noise_variance/||M−S||²`. For skin inversion, observation noise is amplified by `1/(1−a)` and M error by `a/(1−a)`. Limiting the denominator prevents numerical failure; it does not create missing evidence. A brush support is not a measured opacity, and product dose/thickness is not generally alpha either.

This distinction matches natural matting: local color assumptions and user constraints can make a chosen estimation problem solvable without proving that hidden physical colors were uniquely observed. [Levin, Lischinski and Weiss, closed-form matting](https://www.ee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf).

The physical mismatch is additional to algebraic ambiguity. A cosmetic can absorb, scatter, fill surface relief, alter specular reflection and interact with underlying skin. Camera RGB integrates spectra and may include nonlinear tone mapping, clipping and unknown white balance. A single scalar alpha and one spatially constant M need not represent those effects, even if a numerical fit is excellent.

### Why attenuation is the better contract

Under the ideal mixture, reducing alpha by a fraction t gives:

`O=(1−(1−t)a)S+(1−t)aM = I+t(S−I)`.

The desired edit still depends on unknown S. Calling t “attenuation strength” does not magically identify skin. The defensible alternative is to declare a **reference-conditioned appearance target R**, limit the supported change toward R, and retain I whenever credible references disagree. R is not renamed “recovered S.”

Separate three quantities in any future prototype: an externally accepted edit support, an appearance adjustment, and uncertainty/abstention reasons. Do not use the present alpha hint interchangeably for all three.

## 4. Primary sources that change the decision

The stopping criterion is reached: additional general makeup-transfer/beauty papers would not alter the evidence requirements. No further literature search was made on resumption.

| Primary work | What it contributes to P4 | What it does not establish |
|---|---|---|
| [Levin et al., A Closed-Form Solution to Natural Image Matting, CVPR 2006](https://www.ee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf) | Separates conditional solvability under local smoothness/user constraints from the unconstrained layer ambiguity | Authentic skin under opaque cosmetics |
| [Ojima, Tsumura et al., Measurements of Skin Chromophores by ICA and the Application to Cosmetics, 2003](https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8587.pdf) | Polarized-light separation of surface/body reflection precedes pigment analysis, under optical-density and independent-pigment assumptions; useful evidence for constrained biological-color analysis | Cosmetic-layer removal: its cosmetic evaluation concerns skin-product effects, not peeling unknown makeup off RGB. Nor does it validate calibrated concentrations from Retouch's intentionally approximate basis |
| [Li, Zhou and Lin, Simulating Makeup through Physics-based Manipulation of Intrinsic Image Layers, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_Simulating_Makeup_Through_2015_CVPR_paper.pdf) | Models makeup through albedo, diffuse shading and specular layers using cosmetic properties; supports separating categories and finish from color | Inverting an unknown product from one image. It is principally a forward simulation method; its category simplifications are not universal rules for every formulation |
| [Lanza et al., Practical Appearance Model for Foundation Cosmetics, 2024](https://diglib.eg.org/items/679aea5c-9968-4515-b63c-15968f358bf9) | Measured-reflectance validation of layered scattering with diffusers and platelets explains why matte, glossy and velvety foundation require more than alpha-color blending | A deployable single-RGB makeup remover or unique inverse of that richer material model |
| [Yang, Taketomi and Kanamori, Makeup Extraction of 3D Representation via Illumination-Aware Image Decomposition, Eurographics 2023](https://yangxingchao.github.io/makeup-extract-page/) | Directly relevant learned/inverse-rendering challenger: bare/makeup/alpha extraction after geometry, lighting and material estimation | Observational ground truth for its bare-skin output. Strong 3D priors and completion supply information absent from the photograph |
| [Chong et al., MicroGlam, SIGGRAPH Asia 2023 Technical Communications / 2024 arXiv posting](https://arxiv.org/html/2401.05339v1) | Paired, product-specific capture is a substantially better physical test than unrelated before/after beauty images | Face-wide or all-category validity: it uses microscopic hand patches, nine participants, three products, and manual homography alignment |

The [Yang author implementation](https://github.com/YangXingchao/makeup-extract) is public but depends on CUDA/PyTorch, nvdiffrast and prerequisite face models. Its published UV material collection is algorithmically extracted, not a captured bare-skin/opacity truth set. It is a bounded **second-round challenger** only: use its suggestion as an uncertain local edit direction, hold source geometry/detail and accepted supports fixed, reject invalid mappings, and apply the same output bounds. Do not import completed facial texture as recovered observation or infer production suitability from a demo. Model/data access and rights must be checked before any use.

## 5. Compare the five approaches

| Approach | Conditional usefulness | Main failure | Recommended role |
|---|---|---|---|
| Current linear mixture | Cheap conditional alpha utility when endpoints are known; existing heuristic baseline for compact appearance anomalies | Mixed observations seed M; single M cannot represent multiple cosmetics; S is prior-filled and algebraically adjusted; alpha/skin are non-identifiable | Diagnostic baseline, not product truth |
| Local-reference constrained estimation | Nearby, accepted bare skin under comparable illumination can bound a local appearance target; several references can reveal uncertainty | Foundation may cover every possible reference; cheek/forehead/eyelid color and lighting differ; local references do not reveal buried marks | **First candidate** for confirmed, translucent, small-area color attenuation; abstain if reference estimates disagree |
| Intrinsic/chromophore assistance | Helps ask whether a proposed edit is more consistent with pigment, shading or highlight variation | Blush versus natural redness, contour versus shadow, and cosmetic pigments versus skin pigments remain confounded; Retouch's current basis is approximate | Auxiliary consistency/abstention signal, not sole material label or physical inverse |
| Spatially regularized/multiscale decomposition | Stabilizes a supported appearance field and can separate broad coverage variation from narrow boundaries | Both natural shading and cosmetics are smooth; priors may move edges, flatten real texture or spread edits across protected boundaries | Add only after local references work; regularize the correction field, with no hidden texture synthesis |
| Bounded learned estimator | May suggest useful targets in familiar, translucent makeup regimes | Dataset priors, skin-tone bias, alignment, occlusion completion and out-of-distribution cosplay; believable output is not authentic hidden skin | Optional second round, never automatic replacement or first experiment prerequisite |

A posterior, ensemble or regularizer can narrow uncertainty **conditional on its assumptions**. Report that dependence. An optimizer's convergence, a low forward residual, a realistic face or a model's confidence cannot by itself establish recoverability.

## 6. Category decisions

These are conservative product-policy deductions from the inverse problem and the cited optical evidence, not an experimentally validated classifier. A category label alone does not establish opacity, finish, intent or a trustworthy reference.

| Category | What may remain observable | Default and explicitly permitted scope |
|---|---|---|
| **Foundation / concealer** | Thin foundation may retain photographed texture, but its broad color/finish contribution has no bare reference when the face is covered. Concealer can deliberately hide marks and discoloration. | **Abstain from automatic unmixing/removal.** Allow separately authorized, bounded coverage/finish repair. Consider local translucent attenuation only with credible nearby bare reference. Do not recreate hidden pores, freckles or blemishes. |
| **Blush** | Diffuse localized color can sometimes be compared with adjacent accepted skin; it can also be indistinguishable from natural redness or mixed lighting. | Preserve by default. **Best first attenuation case** only with confirmed makeup support, user intent, matte/translucent appearance and reference agreement. Unknown redness => abstain. |
| **Contour** | A spatial darkening field is visible, but product, anatomy and illumination can produce the same signal. | **Abstain from automatic removal.** Explicit style attenuation may be offered as tone editing, not restored geometry/shading or recovered skin. Keep this distinct from chroma-only blush attenuation. |
| **Eyeshadow** | Some color and finish can be modified without recovering the eyelid beneath it; layering, lashes, fold shadows and shimmer cause ambiguity. | **Protect by default.** Separate, explicitly selected eyelid-color editing may attenuate an observed look; no lash/fold reconstruction and no generic face-skin reference fill. |
| **Eyeliner** | The painted shape is observable; occluded lid/lash detail is not. | **Protect. Automatic unmixing abstains.** Removal is a separate reconstruction/inpainting request, not P4 recovery. |
| **Lipstick** | Surface lip texture may remain, but native lip pigment/finish is not determined by the painted color. Skin-chromophore priors are not an adequate lip substrate model. | **Protect.** Optional lip-specific color/finish attenuation requires explicit intent and maintains observed boundaries/detail; do not claim authentic bare lip color. |
| **Cosplay face paint** | Intended pattern and coating defects are visible; opaque regions erase underlying appearance. | **Protect design and boundaries.** A separate owner-approved coverage/crack repair operation may edit the coating. Opaque paint removal => automatic abstention. Translucent local paint needs the same evidence as other attenuation, not a blanket exemption. |
| **Glitter / specular cosmetics** | Directional sparkles and highlights reflect product, illumination and geometry; saturation may destroy measurements. | **Protect intentional sparkle. Abstain from unmixing.** A separately requested shine/finish edit is appearance control; clipped/occluded skin cannot be recovered. |

Always preserve accepted identity marks, facial hair and anatomical boundaries regardless of cosmetic category, unless the owner explicitly chooses another operation. A five-level RGB cap is not permission to edit a mole or eyeliner.

## 7. First controlled experiment for Claude

**Build an offline identifiability-and-attenuation harness, not another automatic makeup detector and not production integration.** Proposed artifact: `scripts/qa/p4_identifiability_experiment.py`, a reviewed manifest, focused tests and a report. No new files of that kind are implemented in this research turn.

### Stage A — minimum decisive mathematical controls

Use deterministic arrays and exact external supports, not labels inferred by the current outlier detector:

1. **Oracle case:** known S and M, varied alpha and noise. Test conditional projection and recovery against supplied truth. Label this extra-information oracle separately from all single-image arms.
2. **Same-observation counterexample:** supply identical I/supports but two distinct valid S/M/alpha explanations. A simple grayscale example replicated in RGB is `I=.55`: `a=.5,S=.45,M=.65` and `a=.5,S=.35,M=.75`. Both are smooth, valid and indistinguishable to the estimator. Add a spatial variant with different hidden detail. A single deterministic estimate must not claim to have identified both truths.
3. **Opaque control:** alpha=1 with different hidden skin/marks. Required result is non-identifiability/abstention, not a guessed face or a falsely confident S.
4. **Confounds and conditioning:** nearly equal endpoints; shadows and natural-redness fields with zero makeup; multi-color cosmetic regions; noise/clipping; protected edges. Include accepted-support boundaries and caking-only edits to test the inspected protection risks.

Reuse the existing P4 synthetic constructions where helpful, but not their automatic detector as the experimental oracle. Keep the synthetic compositing domain explicit: encoded-BGR model tests validate the current algebra; a separate linear-light track tests a radiometric mixture. Never present success on encoded synthetic composites as validation of cosmetic optics.

### Stage B — smallest useful physical pilot

Two known participants, **one development subject and one untouched check subject**, suffice for a mechanism pilot, not for a production-performance claim. For each subject select four small, reviewed supports: translucent blush, thin foundation, an opaque concealer/paint region, and a specular/highlighter region. Capture the same location before makeup, a repeat untouched baseline, a light application and a heavier application: **32 patch-state observations**. Different states may come from shared full-face captures; they are not 32 independent subjects or necessarily 32 separate shutter events.

Use fixed native camera geometry, exposure, white balance, color pipeline and lighting; retain source files and accepted registration/validity masks. The repeat baseline measures noise, registration and physiological drift. Do not use post-removal redness as the bare-skin reference. If a registration uncertainty exceeds the feature scale, abstain from feature-level recovery scoring. After the minimum pilot, add changed-light/no-product controls and all protected makeup categories before any integration proposal.

Only the makeup observation and approved **same-observation local references** reach ordinary estimators. Before-makeup/light-application images are scorer-only truth. Do not leak them into S initialization. Applied product amount is metadata, not ground-truth alpha. A lighter photographed application is evidence for an appearance-reduction target, not proof that halving alpha models that material.

If new capture is unavailable, a verified subset of the [MicroGlam author data](https://github.com/tobyclh/MicroGlam) may support a paired hand-patch mechanism check after access/rights and alignment review. It cannot fill missing face, eyeliner/lip, opaque-concealer or controlled-dose evidence. Existing unpaired cosplay portraits and algorithmically demade-up datasets cannot supply the missing bare-skin target. If no valid paired data are available, finish Stage A and explicitly stop before a practical quality verdict.

### Arms and fixed factors

- **A0:** unchanged I / abstention.
- **A1:** present linear-fit heuristic, initialized using an external support prior. The initialization is declared heuristic, not a measured opacity; no automatic candidate generation. Separately retain the current product entry only as a regression reference, not as a fair detector-free inverse-quality arm.
- **A2:** local-reference bounded appearance attenuation, no alpha/S/M truth claim.
- **A3:** A2 plus intrinsic/chromophore consistency-based refusal; no new physical concentration claim.
- **A4:** A2 plus spatially regularized/multiscale correction field.
- **Oracle:** independently supplied S/M, mathematical controls only. No learned model in the first run; Yang's method remains optional second-round work.

Keep image encoding, external allow/protect masks, local-reference regions, crop/scale, adjustment strength and output bounds identical across A1–A4. Compare the **same intended operation**, not current coverage smoothing against another arm's complete makeup removal. For the first translucent-color comparison, bound a low-frequency chromatic correction toward each arm's target and retain source luminance/detail; score contour/finish changes in a separate operation, not as failures to remove color.

Use a predeclared small strength (for example .25 of the proposed appearance correction) and a common encoded-channel bound no greater than the current ±5 for the initial diagnostic. These are conservative experimental settings, not universal perceptual safety thresholds. Reapply external protection at final composition; record both proposed and applied deltas so a cap or zero-coverage arm cannot masquerade as accurate estimation. Vary local references on development data to expose ambiguity; lock all choices and thresholds before the untouched subject. Do not optimize against its paired truth.

### Scores that distinguish evidence from plausible reconstruction

1. **Latent truth versus fit:** on synthetic controls, report S and alpha error separately from `||I−recompose(S,M,a)||`. On real captures, do not report physical alpha accuracy without an independent opacity measurement.
2. **Observable recovery:** measure local color error to captured bare/lighter states, and source-observable pore/mark/hair contrast, position and width within registered valid regions. Do not reward fabricated hidden detail or a smoother-looking surface.
3. **Useful partial attenuation:** quantify whether the proposed change moves toward the photographed lighter application, beyond repeat-baseline uncertainty, without overshooting or altering protected structure. Intermediate linear interpolations are style targets, not measured intermediate cosmetics.
4. **Harm:** maximum and changed area inside protected supports; boundary halo/edge displacement; native texture loss; luminance/chroma drift on untouched and shadow/redness controls. Separate appearance preference from these errors.
5. **Uncertainty and coverage:** abstention by category and reason, eligible area, reference-to-reference variability, and error among accepted edits. An ensemble agreement is not calibrated uncertainty until checked on held-out measured targets.
6. **Diagnostics:** native I/reference/target/output crops; signed proposed/applied changes; alpha clearly labelled “prior/fit”; alternative feasible decompositions; protected boundaries; profiles and per-case failure tables. Blind owner appearance review is supplementary, not the latent-truth metric.

A result is not positive merely because it beats the current heuristic or has a low reconstruction residual. It must improve a measured, observable target beyond baseline repeatability while retaining the safety constraints. A one-subject check does not establish population fairness or an integration-ready error rate.

## 8. What could falsify or narrow the recommendation?

- **Universal single-image unmixing:** the exact non-uniqueness counterexample cannot be refuted by attractive outputs or average benchmark scores. Recoverability requires additional observations or explicitly restricted priors. Adding calibrated product/lighting measurements changes the problem; it does not solve the original unconstrained one.
- **A restricted unmixing claim could become justified:** independently anchored endpoints/materials and registered bare-skin captures could show stable, accurate recovery for a defined translucent category, camera/lighting regime and opacity range. It must beat identity and local-reference baselines, preserve real identity detail, report failures/uncertainty, and generalize to untouched subjects/products. If physical alpha is advertised, validate it independently rather than against a brush mask or a network's pseudo-label.
- **Bounded attenuation could fail:** if its held-out accepted edits do not improve the measured target beyond repeatability, if changing credible references changes the correction direction, or if natural redness/shadows/identity marks are harmed despite the bound, reject that category/operation and retain abstention. This recommendation does not promise that attenuation will work.
- **Category protection could be relaxed only individually:** show reliable material/intent support and acceptable measured harm for that category under explicit owner authorization. This can justify a style edit; it cannot reveal pixels destroyed by opacity or clipping.

**Final decision:** reframe P4 as bounded, reference-conditioned makeup-appearance attenuation; split attenuation from coverage/finish repair and from eye/lip/paint-specific edits; automatically abstain when the inverse is non-identifiable. Claude's next implementation should be the offline Stage-A/Stage-B harness above, with no production wiring and no generic makeup-free-face generator.

## 9. Production follow-through (2026-09-07)

The accepted policy is now represented by the isolated
`apply_bounded_makeup_attenuation(...)` leaf in
[`retouch/makeup_unmix.py`](../../retouch/makeup_unmix.py). It requires an
externally reviewed support mask, a separately supplied reference image/mask,
and an explicit category. The first supported category is confirmed
translucent blush/color attenuation; unsupported or protected categories
(including lipstick, eyeliner/lashes, eyeshadow, face paint, glitter and
unknown intent) abstain. The operation shifts only OKLab chroma toward the
reference, preserves source luminance/detail, restores all pixels outside the
eligible support exactly, and enforces finite/range checks plus a maximum
encoded-channel delta of five DN. Its `BoundedAttenuationResult` records
applied/abstained state, reason, support coverage and proposed/applied bounds.
`RetouchEngine.apply_bounded_makeup_attenuation(...)` is the public Engine
adapter for the same leaf; it remains an explicit operation and is not an
automatic pipeline stage.

The pre-existing `apply_makeup_unmix` / `apply_makeup_coverage_even` path is
retained only for backward compatibility and remains unchanged in the default
recipes. It is explicitly documented as a legacy heuristic, not a physical
unmix or bare-skin recovery claim; no new caller, recipe or GUI control routes
through it. The bounded leaf is not automatically wired into the pipeline,
because P4 does not establish automatic support generation or a general
single-image inverse.

Focused tests cover bounds, zero strength, protection, missing/invalid supports,
abstention, determinism and legacy behavior. No production claim is made for
hidden texture, recovered identity marks or categories outside the stated
reference-conditioned blush operation.
