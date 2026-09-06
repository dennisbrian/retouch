# FA-02: photographed micro-texture representations after smoothing

Date: 2026-09-06. Research and experiment design only. No prototype, production change, new portrait render, or native-resolution representation bake-off was performed in this tranche.

Continues [FA-02 residual-restoration evidence](RESEARCH_FA02_MICRO_TEXTURE_RESTORE_2026_09_05.md). FA-01 is closed for this task; FA-03 classification is being handled separately. Accepted correction/protection supports are external inputs throughout this proposal.

## Recommendation

Test **source-coordinate, explicitly band-limited residual restoration first**, then its multiscale extension. Restore the difference between source and processed coefficients in selected bands, rather than adding the source high-pass band again. Keep the processed low-frequency base. Retain a disabled control and current raw residual at the same gain and eligibility support. Add an orientation-aware arm to test fine-hair retention and edge contamination, without treating orientation as permission to edit.

Ranking is by suitability for a controlled first experiment, not measured image quality:

1. One selected DoG band of the source-minus-processed signal: smallest interpretable challenger.
2. Selectable multiscale Gaussian/Laplacian band differences: recommended general representation if the single-band result earns further work.
3. The multiscale representation with orientation/coherence and noise-reliability weights: conditional challenger for fine hair and edges.
4. Stationary wavelet subbands with conservative coefficient reliability/shrinkage: useful alternative for correlated noise, with more choices and memory.
5. Local Laplacian detail remapping: strong edge-aware literature, but a more nonlinear and expensive experiment with harder attribution.
6. Current raw residual: retain as the baseline; it has no mechanism for isolating useful photographed texture.

Restoration disabled is an essential control and can remain the appropriate outcome for unresolved/noisy/soft faces. It is not a failed benchmark arm. Reusing Retouch's existing low/mid/high decomposition is a low-cost engineering comparator alongside ranks 1–2, but its wide high band cannot separate pores from noise.

No representation alone identifies whether a same-scale feature is a pore, freckle, blemish, eyeliner, hair, or noise. The question here is which representation exposes useful scale/orientation controls while retaining source correspondence and avoiding unnecessary broad corrections.

## 1. Frozen code rebaseline from this session

The code inspection was completed before the interruption and was not repeated on resume. Observed HEAD: `710cb5b8ee61c14ee38bf7b861146c90fea2e03c`, with existing changes in params, recipes, recipe tests, docs index and the execution plan. Those changes were left alone. References below describe the inspected implementation rather than the status labels in the shared execution plan.

[`SkinProcessor.restore_micro_texture()`](../../retouch/skin.py#L985) computes, in existing BGR code-value space:

`O = clip(S + a M (X - S), 0, 255)`

where `X=pre_smooth_canvas`, `S=canvas immediately before this call`, `a=(strength/100)*smooth_strength`, and `M` is the dimensional mask. Default function arguments 20 and .5 give `a=.10`. The function performs no color-space linearization, frequency decomposition, noise estimation, edge rejection or mark lookup. Float32 input stays float32; the integer path casts to uint8 after clipping. No-op guards are nonpositive strength/smoothing, absent regions, or mask maximum below .01. There is no internal clamp of positive strength to the documented range.

`_build_dimensional_mask()` (`skin.py:103`) accumulates the supplied soft region masks with **clipped addition**, then Gaussian-feathers them. This is not pixelwise max when soft masks overlap. The nine fields are nose bridge, two cheek highlights, two under-eyes, two eyes and two crow's-foot regions. The feather kernel is `max(3, int(min(H,W)*.01)) | 1`. The helper does not intersect this result with skin or correction/protection supports. Its name does not make the restored signal a texture band.

With valid masks and `0<=aM<=1`, this is a local interpolation toward X: `(1-aM)S+aMX`. It can restore every changed channel and scale. A constant tone difference of +10 code values is restored as +1 at full mask with a=.10, despite containing no micro-texture. It can also undo a brightening correction because the residual is signed. This is an algebraic consequence, not a new image-quality experiment.

### Snapshot provenance matters

The ordering already inspected in [`perf_optimizations.py`](../../retouch/perf_optimizations.py#L440) is:

`makeup coverage / albedo conditioning → snapshot X → frequency combine → optional under-eye shadow smoothing → optional freckle healing → optional exposure lock → optional flatten → snapshot S / micro restore → later skin operations and general blemish removal`.

Therefore:

- X is not necessarily the pristine camera image; makeup/albedo conditioning can precede it.
- X−S contains the combined differences of **all** stages between the snapshots, not just smoothing.
- Optional freckle healing occurs before restoration. An actually healed freckle can therefore be partially restored. General `blemish.remove()` occurs later and cannot be resurrected by an earlier call in the current ordering.
- The earlier note's broad “nothing has been repaired yet” wording is too strong; its narrower statement about general blemish removal remains correct.
- `ParamSpec.micro_restore` defaults to 20. Four explicit recipe assignments do not establish that only four effective recipe paths can restore texture; effective parameter resolution must be recorded in a future experiment.
- The earlier `defect_excluded` arm changed the support mask, **not frequency bands**. It is evidence about spatial exclusion, not evidence comparing texture representations.

The previous measured defect-contrast recovery and failed residual-magnitude discriminator remain prior evidence, not new results here. Neither their particular marks nor whole-face high-frequency energy establish useful pore recovery.

### Frequency separation already present

[`FrequencySeparator.separate()`](../../retouch/frequency.py#L626) creates full-resolution float layers from a uint8 snapshot:

`low = G_large(X)`, `mid = G_medium(X) - G_large(X)`, `high = X - G_medium(X)`.

Kernel widths scale with face width: factors .12 and .04, minima 5 and 3. These are Gaussian kernel dimensions, not literal pore sizes or measured frequency cutoffs. `low+mid+high` reconstructs that input before clipping/casting.

`combine()` can attenuate mid/high, alter a broader blotch band, and smooth low+original-mid using guided, bilateral or anisotropic engines. The final composition uses masks and can spread changes across nominal bands. It also uses high-band energy for smoothing adaptation, so that existing adaptation must be frozen in a representation comparison. Choosing the anisotropic smoother changes S; it does not by itself change the later residual representation.

`pore_synthesis` adds seeded high-pass random noise. That is not photographed pore recovery and is disabled in the proposed representation experiment. Reusing source high/mid buffers can save work, but adding `high(X)` again would duplicate detail already surviving in S. A fair recovery comparator uses `high(X)-high(S)` (or equivalent selected differences), with a documented float/uint8 boundary.

## 2. Representation versus eligibility

For a linear analysis/synthesis pair T/R, define source and processed coefficients `x_k=T_k(X)`, `s_k=T_k(S)`, and missing detail `d_k=x_k-s_k`. Recover only selected detail channels:

`O = S + a * M_final * sum_k R_k(g_k * q_k * e_k * d_k)`.

`g_k` chooses bands/gains. `q_k` expresses coefficient reliability/noise evidence. `e_k` is eligibility transferred from **externally accepted** supports to the coefficient footprint. `M_final` enforces allowed output pixels. Low-pass/DC coefficients have zero recovery gain. This expression is a design contract, not an implemented API. For identity/noise checks, retain signed coefficients and their original coordinates; do not retain magnitude alone or reconstruct with new/random phase.

For a fixed linear transform, `T(X−S)=T(X)−T(S)`, allowing a cheap residual-only implementation. This identity does not hold for independently adaptive nonlinear filters or independently noise-shrunk images. Freeze their guidance or explicitly compare source/processed decompositions. Noise shrinkage needs source/noise evidence; a large residual is not itself evidence of useful lost texture.

External support semantics remain explicit: corrected defect pixels, regions whose appearance must not change, and pixels eligible for texture restoration are different inputs. A preserved freckle is not automatically a defect. No detector or new mark classifier is proposed.

### Two leakage mechanisms to measure

1. **Analysis/synthesis footprint leakage:** a corrected spot or eyeliner edge contributes to neighboring filter coefficients. Zeroing its output pixels alone does not prevent a surrounding ring. Propagate accepted exclusions through each band's full analysis+synthesis footprint and taper recovery inward into the remaining valid region. Track the resulting lost-texture margin. For finite separable DoG filters this footprint can be bounded exactly; pyramids/wavelets require their accumulated footprint. Do not choose a universal dilation radius.
2. **Spatial modulation leakage:** multiplying a band by a spatial mask broadens its spectrum. Gaussian/DoG bands also have smooth frequency tails, not brick-wall cutoffs. “Band-limited restoration” here means explicitly scale-limited extraction with measured post-composite leakage, not an assertion that a spatially varying final result has perfectly zero energy outside those bands.

Do not zero a hard hole in X before filtering: that creates a new artificial edge. Mask-normalized convolution or replacing contaminated analysis pixels would be a separate extraction ablation, not an unnoticed change to the first experiment. Avoid patch borrowing into corrected supports; uncertainty about the original texture is not permission to manufacture it.

## 3. Comparison against the requested failure modes

These are expected behaviors derived from the representations and literature, not measured Retouch outcomes. All candidates receive the same external supports and gain in the comparison.

| Representation | Restored blemish/freckle | Restored eyeliner/makeup edge | Sensor/JPEG noise | Broad tone/shading in texture |
|---|---|---|---|---|
| Raw X−S | Returns any corrected difference | Returns altered edge differences | Returns removed noise/artifacts; typically amplification relative to S | Directly returns it, including DC/chroma differences |
| Single DoG band | Returns same-band spot detail; cannot identify type | Step edges occupy the band and produce lobes | Finest exclusion helps only if useful signal and noise separate by scale | DC rejected; broad content attenuated, not perfectly absent |
| Selectable Gaussian/Laplacian multiscale | Can omit contaminated scales locally; fine spot edges survive other bands | Edge contributions span levels; support-aware gating still needed | Can disable/downweight noisy levels; not a noise model by itself | Keep S's coarse base; coarsest selected levels and masks still leak |
| Wavelet subbands | Spots activate several scales/orientations | Strong directional coefficients include eyeliner | Neighborhood/cross-scale noise estimation is natural; compression is structured and may survive | Omit approximation band; coarse wavelets and boundaries still carry shading |
| Local Laplacian | Detail/edge remapping is not defect recognition | Can keep strong edges unchanged in its intended operation; subtracting independently filtered X/S needs care | Small-amplitude enhancement can amplify noise/JPEG; explicit coring is required | Better separation of large transitions from detail, but nonlinear scale mixing and guidance changes remain |
| Existing low/mid/high | Broad mid/high categories mix spots and texture | Existing high includes cosmetic edges | High includes noise; high-band energy adaptation cannot certify pores | Excluding low helps; broad mid and mask effects remain |
| Oriented/steerable bands or tensor-weighted multiscale | Round pores and round defects remain inseparable | Directional structure can be exposed; hair and makeup can share it | Cross-scale/neighbor agreement can help, but correlated noise/block edges are structured | Coarse bands still need omission; orientation alone removes no DC |
| Source-coordinate residual gating | Inherits the chosen transform; coordinates do not reject defects | Inherits transform and external eligibility | Source noise also has exact coordinates | Inherits band/base choice |
| Disabled | No reintroduction by this stage | No reintroduction by this stage | No added noise by this stage | No reversal by this stage |

| Representation | Repeated/copy-like texture | Halo risk | Real pores / fine hair | Small, soft or high-ISO faces |
|---|---|---|---|---|
| Raw X−S | No patch repetition mechanism; can restore source patterns/artifacts | Existing halos can return; soft mask can expose tone transitions | Returns genuine removed signal together with everything else | Can revive noise rather than unresolved texture |
| Single DoG | No copying; retain signed phase | Positive/negative edge lobes plus masking | Simple, but a narrow band can miss pore sizes/hair widths | Useful/noise spectra overlap; appropriate result may be no restoration |
| Multiscale Gaussian/Laplacian | No copying with source-aligned coefficients | Unequal gains around edges create rings; decimation adds phase concerns | Best control over several observed scales without synthesizing detail | Finest reliable level may shift/disappear; test crop/resize stability |
| Stationary wavelets | No copying; aggressive thresholding can create patterned remnants | Threshold ringing; critically sampled variants add shift sensitivity | Directional/scale control; can erase weak real hairs/pores | More stable under translation than critically sampled coefficients, but cannot undo inadequate sampling |
| Local Laplacian | No patch copy required, but remapping changes appearance | Literature specifically addresses edge halos; external exclusions can reintroduce them | Strong detail control; amplitude-based “detail” includes defects and noise | Range thresholds need signal/noise context; excessive noise is a documented weakness |
| Existing low/mid/high | No copy unless synthesis/transfer is enabled | Mask and broad-band blending can create transitions | Wide high band preserves some fine signal, without enough selection | Face-scaled widths do not account for focus/noise or true optical resolution |
| Orientation-aware variant | No copy when coefficients stay at their source locations | Directional masks can produce streaks or anisotropic rings | Potential advantage for resolved hair; do not reject isotropic pores for low coherence | Orientation is unreliable in flat/noisy areas; gradients alone overstate reliability |
| Source-coordinate residual gating | Explicitly prevents donor-patch repetition; phase/registration errors can still double edges | Validity boundaries and coefficient footprints matter | Faithful only where source signal is observable and corresponds to S | Geometry/blur mismatch can require abstention |
| Disabled | None added | None added | Leaves smoothing's losses in place | Most conservative added-artifact control; not automatically best appearance |

### Specific representation choices

**DoG / band-pass:** use `B_k(Z)=G_sigma_k(Z)−G_sigma_(k+1)(Z)`. The finest retained Gaussian suppresses some pixel-scale noise; the broader Gaussian removes coarse content. A single LoG response is a scale-normalized derivative diagnostic, not a replacement pixel residual: adding arbitrary LoG values without a defined reconstruction changes edge/contrast behavior. FFT brick-wall masks are not the first choice because of global support and ringing.

**Multiscale:** a full-resolution telescoping Gaussian stack makes scale selection and original-coordinate masks easiest to audit. It is not a decimated Laplacian pyramid. Include a proper Laplacian pyramid as a storage/runtime challenger, with correct expansion/reconstruction and one-pixel-shift tests. Low-pass base and finest residual are separately switchable. Adjacent DoG bands with identical gains telescope to a wider single DoG; that setting is not evidence of a distinct multiscale advantage. Multiscale earns its complexity only through useful unequal, noncontiguous or locally justified band selections.

**Wavelets:** start with a stationary/undecimated transform if testing them, to reduce translation sensitivity. Standard decimated LH/HL/HH coefficients are efficient but represent oblique fine hairs awkwardly and are sensitive to phase under coefficient selection. Wavelet labels are analysis directions, not mark classes. Conservative continuous shrinkage can use a noise estimate; blind magnitude thresholding also removes weak photographed pores.

**Orientation:** retain scalar isotropic bands for pores and test an additional directional fine-hair path. Do not sum overlapping filter responses without a reconstruction/normalization rule. A structure-tensor weight is cheaper than a full steerable transform; it is a modifier, not a new independent texture source. Existing `_compute_orientation_field()` reports an angle and gradient-magnitude-derived confidence, not an eigenvalue-coherence/noise probability. A future experiment should distinguish these quantities explicitly.

**Source-coordinate transfer:** keep X/S on the same pixel grid and copy no donor patches. If a future earlier geometry operation changes coordinates, a known validated warp and validity map are required; interpolated coefficients, occlusions and misalignment get separate diagnostics. Do not silently substitute neighboring cheek texture or a texture library when source pores are missing. Corrected regions with no trustworthy source detail stay without restoration in this experiment.

## 4. Relevant primary literature

Sources were checked during this session. Claims about applying them to Retouch are hypotheses. None establishes native-resolution pore preservation for this corpus.

| Source | Direct relevance and limit |
|---|---|
| Burt & Adelson (1983), [The Laplacian Pyramid as a Compact Image Code](https://persci.mit.edu/pub_abstracts/pyramid83_abs.html) | Localized multiscale difference representation and reconstruction. Supports separate scale controls, not semantic separation of noise, makeup and pores. |
| Paris, Hasinoff & Kautz (2011), [Local Laplacian Filters](https://people.csail.mit.edu/sparis/publi/2011/siggraph/Paris_11_Local_Laplacian_Filters_lowres.pdf) | Distinguishes detail from large transitions through local remapping. §5.2 explicitly discusses amplification of noise and lossy-compression artifacts; §5.4 notes failure under excessive noise. Useful edge-aware challenger, not a reason to trust every recovered detail. Luminance versus RGB processing also changes color-contrast behavior. |
| Aubry et al. (2014), [Fast Local Laplacian Filters: Theory and Applications](https://imagine.enpc.fr/~aubrym/projects/llf/index.html) | Describes acceleration and relationship to bilateral filtering/anisotropic diffusion. Supports keeping local Laplacian as a costlier conditional arm. Published speedups are relative to their original implementation, not Retouch CPU timings. |
| Simoncelli & Freeman (1995), [Steerable Pyramid, author resource](https://www.cns.nyu.edu/~eero/steerpyr/) | Multi-scale oriented representation, reconstruction and translation/rotation behavior. Directly relevant to preserving oblique fine hair and avoiding orientation bias. It offers no makeup/fine-hair semantic guarantee. |
| Portilla, Strela, Wainwright & Simoncelli (2003), [Image Denoising Using Scale Mixtures of Gaussians in the Wavelet Domain](https://www.cns.nyu.edu/pub/lcv/portilla03-reprint.pdf) | Estimates coefficient signal using spatial, orientation and cross-scale neighborhoods plus noise covariance. This is stronger noise evidence than whole-image HF energy. The paper treats Gaussian noise, including known covariance, and explicitly notes camera noise's correlation and signal dependence; JPEG requires separate modeling/controls. No assumption that the largest coefficients are authentic pores. |
| Buades, Martorell & Sánchez-Beeckman (2024), [Joint Denoising and HDR for RAW Image Sequences](https://repositori.uib.cat/items/1cb00676-7488-4dd3-b274-63066a9007ac) | Joint denoising/fusion using selected spatiotemporal patches and weighted PCA. Relevant here to independent-observation evidence: aligned repeated captures can help distinguish scene detail from noise. The university's indexed abstract was available; direct full-text access failed. No burst/HDR pipeline or donor-patch fusion is proposed for FA-02, and existing JPEG portraits are not assumed to provide such references. |
| Jiang et al. (2025), [Self-supervised Texture Filtering](https://doi.org/10.1145/3744899), [author publication page](https://www.zhangqing-home.net/) | Direct structure/texture decomposition research using self-supervision. The target is general texture removal/structure retention, not photographed pore/noise separation after a known edit. Publisher-indexed abstract/intro and author listing were inspected; no runnable model was verified or tested. Keep as a representation challenger only if classical evidence later shows a specific gap. |

The 2024–2026 search did not justify starting a learned full-face retoucher or adding a new model to this experiment. A 2024 MIP filtering acceleration paper was screened but concerns rendering-oriented filter acceleration, not evidence of skin-detail fidelity, and is not shortlisted. Generative filtering/reconstruction, general beauty systems and semantic face parsers are outside this tranche. No 2026 method is promoted merely for recency.

## 5. How to decide which bands contain useful information

Useful texture is **visible, source-corresponding structure that smoothing reduced and that restoration can recover without a larger nuisance increase**. It is not “the band with the most energy.”

1. Review native-source patches first: annotate visible pore centers/supports, fine-hair traces, clean flat skin, makeup edges, accepted corrections and their surrounding rings. Mark a pore/hair patch unresolvable if focus/sampling does not support a trustworthy label. These are observability annotations, not a new classifier.
2. Display signed source, processed and missing-detail bands with the same scales. Identify where the removed signal lies. Some source high-frequency detail may already be intact in S and needs no restoration.
3. Measure recovery of the reviewed structure and nuisance separately for each band. When possible use an independent registered capture; correlation with the same noisy source alone rewards restoring its noise and JPEG grid.
4. Test signal/noise evidence within a band: local noise estimates, independent-noise controls, neighborhood/cross-scale consistency and directional continuity. None is a universal pore test; a correlated artifact can pass several of these checks.
5. Tune a small number of band/gain settings on development images, then lock them. Compare a Pareto curve of pore/hair recovery versus noise, edge/correction contamination and halo cost. Report how many pixels/patches were eligible or abstained so a nearly disabled arm cannot win by hiding its coverage.

A transparent initial diagnostic stack can use sigma values `.6, 1.2, 2.4, 4.8, 9.6` at a reference inter-eye distance of 200 px. Log actual pixel sigmas and scaled values; they are exploratory settings, not universal pore bands. For a continuous Gaussian DoG the peak frequency is `sqrt(log(sigma_b/sigma_a)/(pi²*(sigma_b²−sigma_a²)))`. The four octave differences above peak at roughly 3.9, 7.8, 15.7 and 31.4 px wavelength before discrete/truncated-filter effects. This prevents confusing Gaussian sigma with wavelength. Measure actual discrete transfer functions/impulse responses in the prototype.

Keep `X−G_.6(X)` and the coarse `G_9.6(X)` available as **diagnostics**; do not silently restore them. The finest contains both possible tiny hairs/pores and noise. The coarse component measures broad leakage. For small faces do not shrink filter scales below supported sampling or enlarge the face and call interpolated detail recovered. Face-relative scale, native pixels, focus and noise all affect usefulness.

### Metrics and visual diagnostics

| Question | Metric, denominator and comparison | Diagnostic visualization |
|---|---|---|
| Real pores retained? | One-to-one matching to reviewed resolvable source pores; contrast/profile recovery, center displacement and false added extrema. Report source observability and eligible-pore counts. On a controlled clean reference, also measure local band error. | Source/S/O at 100%, marked pore centers, identical-display-scale signed bands and local profiles. |
| Fine hair retained? | Contrast across annotated hair traces, width, continuity/break length, position/orientation error and new parallel traces. Avoid rewarding sharpened double edges. | Native trace overlays, perpendicular line profiles and orientation/coherence fields. |
| Noise returned? | On controlled pairs, project the *incremental restoration* onto the known added noise; compare outputs from independent noise draws. For real images use flat-patch robust variance, spatial PSD and cross-channel correlation, explicitly as proxies. | Noise-gain maps, local 2D spectra and paired clean/noisy-output residuals. |
| JPEG returned? | Change in 8×8 boundary discontinuity and edge ringing versus the same uncompressed host; include shifted block origins and chroma artifacts. Natural image periodicity alone is not proof of JPEG. | Block-phase-aligned residual tiles and edge profiles. |
| Corrected defect returned? | In externally accepted reduction supports, `(contrast(O)−contrast(S))/(contrast(X)−contrast(S))`, signed and with an epsilon/undefined case. Also mean/max delta and support leakage. Preserved marks are scored separately. | Correction mask plus signed recovery, component/ring profiles and no-restoration control. |
| Makeup/edge contaminated? | Delta relative to the accepted processed edge target, edge location, width, overshoot/undershoot and changed-pixel area inside/outside the edge support. | Source/S/O native edge crops, gradient-normal profiles and contamination heatmap. |
| Broad tone leaked? | Low-pass of O−S, local mean luminance and chroma shift, and response to a known constant/ramp correction. Report by ROI, not only a whole-face average that cancels signed errors. | Low-pass residual and fixed-range luminance/chroma maps. |
| Halo formed? | Signed radial/normal residual at correction and mask boundaries; peak overshoot/undershoot and integrated absolute delta in successive external rings, normalized by original correction contrast where meaningful. | Ring overlays and profiles extending beyond the largest filter footprint. |
| Repetition/instability? | New nonzero-lag autocorrelation peaks or duplicated patches relative to X; shift equivariance after undoing a 1 px translation; changes under native downsampling/blur/noise. | Local autocorrelation, side-by-side phase shifts and patch provenance/validity maps. |

For a controlled clean host C and two independently corrupted inputs `C+n1` and `C+n2`, compare both O and the stage increment O−S. This isolates whether restoration returns nuisance removed by smoothing. A single input-source PSNR or correlation is insufficient because that source contains the same nuisance. Patch appearance preferences and structure metrics remain separate from perceptual similarity scores; no LPIPS/SSIM/HF aggregate certifies pore truth.

## 6. First controlled experiment

### Reuse and minimum corpus

Use the existing five Meitu source anchors—DSCF2306, DSCF2308, DSCF2310, DSCF2362, DSCF2365—as development anchors only, plus seven suitable portraits from the existing local corpus. Target **12 real portraits with about four reviewed native patches each**: pores, fine hair, flat/noisy skin, and an edge/correction boundary. Tags can overlap. Include small-face, soft-focus, high-ISO, directional lighting and profile examples after review; the named five are not assumed to cover them. If a stratum is absent, record the gap rather than assigning a synthetic example a real high-ISO label.

Use roughly six development and six held-out portraits with subject/session grouping overriding counts; all previously inspected anchors and all derivatives of a subject stay on the development side. Add existing-corpus images if necessary for a subject-separated holdout. Reuse prior semi-synthetic DSCF2306 correction construction as a mechanism control, not as evidence of biological pore/mark labels.

For three reviewed clean host patches, define eight separate diagnostic conditions (24 variants): constant tone change, broad gradient change, a known corrected compact spot, a corrected/held edge, independent added-noise trials, JPEG degradation, smaller sampling, and soft focus. The noise condition uses fixed seeds with at least two draws. Exact correction supports and an unchanged reference are known. Synthetic defects are signal controls; real preservation evidence comes from real annotations. This is a pilot for ranking representations, not a powered production-safety study.

### Freeze all other factors

Capture X and S once at the relevant pipeline boundary in a future isolated diagnostic run. Keep smoothing engine, actual parameters, photometric conditioning, freckle/under-eye processing, geometry and masks identical across arms. Record source checksum, crop transform, dimensions, inter-eye distance, profile/encoding, available ISO/exposure and software versions. Run at source resolution, saving float arrays and lossless review crops. Use the actual effective parameters, not a count of explicit recipe fields.

Disable pore synthesis, generated grain, donor texture transfer and extra clarity/sharpening in the controlled setup. Evaluate the isolated restoration stage before final sharpening/JPEG; a later delivery check can hold one export path fixed. Changing the smoother and representation together is not this experiment. The existing anisotropic smoother is not the orientation arm.

Use one shared external eligibility mask and **one common conservative guard covering the largest candidate footprint** for the primary representation comparison. This keeps allowed pixels constant. Then run a separately reported band-specific-guard ablation to quantify recovered coverage versus contamination. Otherwise a representation with a larger guard could look safer merely because it restored fewer pixels.

### Required arms

| Arm | Signal added to the same S |
|---|---|
| A0 disabled | Zero |
| A1 current residual | `a M (X−S)` with the shared eligibility support |
| A2 one band | `a M [B_k(X)−B_k(S)]`, selected on development only |
| A3 multiscale | Selected Gaussian/Laplacian band differences, with coarse base excluded; one-hot and unequal-band-gain ablations |
| A4 orientation-aware | A3 with a bounded directional/coherence modifier plus the same fixed noise-reliability treatment; retain an isotropic pore path |
| A5 reuse comparator | Current low/mid/high linear decomposition, using source-minus-S coefficients, not duplicate source high |

Also retain the **unchanged current function with its actual dimensional mask** as an observational pipeline reference. Do not attribute differences between that reference and A1 to representation: A1 has the shared externally accepted eligibility. Freeze any noise-reliability estimator across A2–A4, and include reliability-off on development to separate transform benefit from denoising benefit. Stationary wavelet and local Laplacian arms are second-round challengers only if the first round exposes a remaining specific limitation.

Choose one modest gain for the main comparison (e.g. a=.10, matching the function-default example) and a small development-only gain curve, then freeze it. Do not normalize outputs to equal global HF energy: that can force a quiet band to amplify noise. Compare at equal gain first and at matched measured pore/hair recovery versus nuisance burden second.

### Expected decisions, not promised outcomes

- If A2 removes broad leakage but gains no verified useful texture over disabled, do not proceed merely because its residual looks detailed.
- If A3's advantage disappears when adjacent equal-gain bands telescope, use the simpler representation.
- A4 earns inclusion only if it improves annotated hair continuity or edge isolation at matched pore recovery; rejecting all oriented structure fails.
- If source reliability is inadequate on small/soft/noisy strata, a declared no-restoration outcome is valid. Report that coverage explicitly.
- Any apparent winner must survive native review on the subject-held-out set, independent nuisance controls and equal eligibility. No current evidence selects a winning band or representation.

## 7. Minimal prototype design and computational cost

Design only; no new executable prototype was added.

One offline QA runner should accept saved `source_before_smoothing`, `processed_before_restore`, source coordinate metadata, fixed eligibility/correction/protection masks and ROI annotations. It should never call `detect_marks()` or change the production pipeline. Separate four interfaces:

`extract_bands(X,S,representation,scale_spec)` → signed bands, reconstruction operator, footprint metadata;

`estimate_band_reliability(source_bands,noise_controls)` → reliability and reason maps;

`map_external_supports(supports,footprints)` → coefficient eligibility and final output guard;

`reconstruct_delta(bands,gains,reliability,eligibility)` → float delta, then composite onto S.

First implement A0/A1 and the linear DoG stack, not a model. Use explicit finite kernels, consistent padding and no filter reset at face-tile boundaries. Crop with a halo covering the maximum footprint; score only the unpadded interior. Reuse linear residual filtering when valid. Preserve the processed base and original coordinate phase. Log band gains, exclusions, observability reasons and actual coverage.

Keep color treatment a separate factor: reproduce A1 in BGR first. Compare band-limited BGR with a luma-only arm that carries S's chroma unchanged through the declared transform. Do not combine a switch to linear-light/log-luminance with a new representation and credit the representation for the result. Log clipping; intermediate hard clipping or nonlinear coefficient remapping changes the intended band behavior.

Artifacts: manifest, X/S and lossless O crops, signed per-band maps at common display scales, reconstruction/impulse checks, validity/guard maps, per-ROI metrics CSV/JSON, blinded native A/B crops and the profiles described above. No synthetic pore generation or external face-model dependency is needed.

N denotes face-crop pixels; J levels; K orientations; k is finite 1D kernel width. These are implementation-derived cost estimates, **not measured milliseconds**:

| Candidate | Expected work | Additional transform storage, float32 |
|---|---|---|
| Raw residual | O(N), pointwise arithmetic plus current mask feather | A few scalar/RGB planes; no transform bank |
| One DoG residual | Two separable Gaussian filters, O(N(k_a+k_b)); same transform can filter X−S once | Roughly 3–5 scalar working planes =12–20 MB per megapixel; BGR is about 3× |
| Full-resolution J-band Gaussian stack | J+1 blurs and differences; O(N sum k_j) directly, streaming/incremental blur possible | Stored J+1 scalar planes; 5 planes=20 MB/MP, plus input/output/work buffers |
| Decimated Laplacian | O(N) for fixed small kernels over shrinking levels; analysis plus expansion | About 4N/3 coefficients per scalar pyramid; reconstruction buffers extra |
| Stationary wavelet | Fixed J filtering/synthesis stages; cost grows with J/filter support | Around (3J+1)N scalar coefficients in a compact retained representation; J=3 gives ~40 MB/MP, implementations may retain more |
| Oriented/tensor variant | Tensor gradients and three moment filters; full steerable bank adds K directional channels per scale or FFT work | Several tensor planes, or orientation-bank storage; higher than scalar bands |
| Local Laplacian | Original optimized method O(N log N), larger constants; accelerated sampled-remapping variants depend on intensity sampling | Multiple pyramids or streamed samples; profile the chosen algorithm rather than transferring paper timings |

One scalar megapixel plane is 4 MB in decimal units; live RGB X/S already consume 24 MB/MP before these extras. Avoid caching both full source and processed pyramids when linear residual filtering suffices. Native large-face crops and concurrent face workers can multiply memory quickly.

The anticipated order is raw < one band < a small scalar multiscale bank < oriented/stationary or nonlinear variants, but implementations and kernel support can change it. Benchmark warm and cold timings, median/p95 ms per megapixel and peak RSS on 0.25, 1 and 4 MP face crops with fixed OpenCV threads. Report extraction, reliability, support propagation and reconstruction separately from existing face detection/parsing. No runtime claim is made from the papers' hardware or previous whole-engine bake-offs.

## 8. Evidence required before production

The [Meitu bake-off](RESEARCH_MEITU_RETOUCH_BAKEOFF_2026_09_04.md) rendered at 2048 px long edge and analyzed at 1600 px. Two competitor exports downsampled 4160×6240 sources to 2731×4096; profiles, export settings and edits were not standardized. Registered/resampled HF summaries cannot certify native pores, and a similarity transform does not correct every local facial reshape. Use those five source images as anchors and competitor images as optional appearance references, not clean micro-texture ground truth or a target HF ratio.

Before proposing production code, require:

1. Exact float reconstruction/identity/zero-gain checks; constant/ramp rejection away from crop boundaries; measured finite-filter response; mask/correction impulse tests; 1 px translation and tile-boundary consistency. These are meaningful future prototype checks, not tests run in this research turn.
2. A frozen, subject-held-out native-resolution comparison showing useful pore/hair recovery over disabled and reduced nuisance versus raw residual, with all eight requested failure modes scored separately. Report local maxima and worst crops, not just means.
3. No new restoration inside externally forbidden supports, and quantified halos/coverage loss outside them. Verify corrected freckles/blemishes and protected identity/makeup as different intended outcomes.
4. Independent-noise/JPEG controls and explicit uncertainty where real noisy JPEGs lack a clean reference. Any claim about genuine high-ISO faces requires genuine examples in the held-out set.
5. Gains/scale choices and rejection criteria fixed before the held-out run; no final-quality winner based only on synthetic marks, energy, PSNR or a rescaled contact sheet. Native owner review is required for weak pores and fine facial hair.
6. CPU/memory measurements and an explicit opt-in/compatibility proposal, followed by narrowly authorized integration tests. This report does not authorize changing current defaults or any parallel FA-03 work.

This resolves the next experiment's representation design, not the image-quality outcome. It permits FA-02 research to proceed with fixed external supports while classification work continues independently. A production replacement remains contingent on those measurements.
