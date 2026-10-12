# P6 — Noise-aware clarity / local contrast

2026-09-07 · research/offline evidence plus explicitly opt-in implementation · production baseline `1b3e341`

## Decision

**Intrinsic amplification is reproducible, but modest at the requested strengths.** The current operator boosts non-structural residuals even in a region with no useful detail. On the controlled 2-DN Gaussian fixture, recipe clarity 4 increases high-pass RMS by 4.03% and high-band energy by 8.31%. On the native DSCF1884 sky ROI, the corresponding increases are 3.52% and 7.18%. This is **not** evidence that the historical severe artifacts have returned, nor that every visible source residual is sensor noise.

**RESEARCH_ONLY for noise-aware candidates; NO-GO for replacing production defaults now.** None of the tested gates simultaneously establishes reliable real-JPEG noise discrimination, preservation of useful detail in noise, and satisfactory edge behavior. The strongest reason to continue is calibrated confidence and local bypass, not changing one global radius, epsilon or strength.

**GO for treating bypass as a valid outcome.** If the intended detail cannot be distinguished from the noise floor, add no clarity there. Preserve the photographed input; do not invent detail or silently denoise it. Global bypass is the fallback when a trustworthy local estimate cannot be formed. This conclusion is now exposed through an explicit caller opt-in; it is not enabled by recipes or the default UI path.

### Implementation follow-through (2026-09-07)

The accepted conclusion has been translated into a narrow, opt-in production API without changing the legacy default. `ProcessingContext.clarity_noise_aware` and the end-of-signature `RetouchEngine.process(..., clarity_noise_aware=...)` override select the confidence-qualified local variance/SNR gate. The implementation uses the predeclared P6 constants (L-channel floor 0.25 DN, variance factor 1.5, 9×9 window), applies the existing global guided-filter decomposition, and restores the exact source pixel wherever the gate abstains. It does not denoise, infer semantic regions, alter candidate generation, or enable itself from a recipe.

The opt-in path is intentionally not a universal replacement: its patch-wide Haar estimate remains uncalibrated for arbitrary camera/JPEG pipelines, and confident pixels use a direct LAB reconstruction rather than the research harness's full-frame RGB delta bridge to avoid a native-frame memory spike. Legacy `_add_clarity`/`_F_add_clarity` dispatch and strength semantics are unchanged. Separate focused tests cover explicit opt-in/default identity, exact flat-region bypass, stochastic attenuation with structural response, uint8/float contracts, invalid-input rejection and context propagation. Informational separate-process samples at recipe strength .04 measured 1920×1080 legacy/opt-in **0.135/0.161 s** and **441.2/448.3 MB**, and 3840×2160 **0.308/0.406 s** and **1,171.1/1,195.8 MB**. The latest native 3879×5850 smoke sample measured **0.890/0.914 s** and **2,670.3/2,882.7 MB** (about **2.7% runtime** and **8.0% peak-RSS** overhead); allocator/startup variation is material, so these are not release benchmarks. The option is not enabled by default and its extra local-variance passes are explicit opt-in cost.

## 1. Historical fixed issue versus current evidence

- Recipe `clarity: 4.0` maps to `strength=0.04` in `_stage_global`. It is not a 4.0-strength units bug. Direct grading preset settings are already fractional and are a separate API.
- `1b3e341` fixed the erroneous float/uint8 clarity dispatch and the actual uint8 LAB round trips in the uint8 implementation. The old incident and intermediate near-black output must not be attributed to this experiment's current guided-filter gain.
- One historical wording qualification matters: `_bgr_f_to_lab_u8_conv` describes an **8-bit-scale convention**, not an 8-bit dtype. Its body was already float-native in `1b3e341^`. Comments describing that helper itself as a uint8 round trip are not an accurate numerical characterization. This report follows the implementation, without changing that earlier document or production comments.
- The prior [clarity quantization investigation](RESEARCH_CLARITY_QUANTIZATION_2026_09_07.md) records source/old-path measurements. Its exact sky crop is not specified sufficiently to reproduce the same numbers. P6 uses explicit new native coordinates; it does not present the old measurements as a fresh rerun.
- Current float RGB↔LAB is still not a perfect identity numerically. A separate conversion-only control is necessary, especially below one DN and near export rounding thresholds. That small remaining error is different from the removed integer-LAB path.

## 2. Current implementations and mathematical characterization

### Global grading operator

Sources: [grading.py](../../retouch/grading.py), `_guided_filter`, `_add_clarity`, `_F_add_clarity`; [utils.py](../../retouch/utils.py), `bgr_f32_to_lab_f32` and its inverse; [engine.py](../../retouch/engine.py), `_stage_global`.

The uint8 entry accepts BGR in 0–255; the float entry accepts float32 BGR in 0–1. Both use OpenCV `COLOR_BGR2LAB` through float helpers. This is display/encoded-BGR processing, not a linear sensor-domain clarity operator, and the operator itself performs no ICC-profile negotiation. The float helper rescales OpenCV L* from 0–100 to `L255` in 0–255 and offsets a*/b* by 128. The filtering variable is `l=L255/255=L*/100`.

Let `Q_r` be the normalized square box filter. For self guidance:

```text
μ = Q_r(l)
v = Q_r(l²) − μ²
a = v / (v + ε)
b = μ − a μ
base = Q_r(a) l + Q_r(b)
d = l − base
l_out = clip(base + (1+s)d, 0, 1) = clip(l + s d, 0, 1)
```

`r=max(int(min(height,width)*0.015),5)`, `ε=0.02`, `s=recipe_clarity/100`. **Despite the variable name and some docstrings, r is the box kernel width `(r,r)`, not a radius producing `(2r+1,2r+1)`.** OpenCV default border handling is used. An even kernel is allowed: the 3879×5850 photograph uses width **58**, with the default even-window anchor. There is no noise floor estimate, semantic gate, confidence measure or explicit midtone weighting.

After modifying L, a/b are retained, subject to inverse-helper bounds. The inverse clips L to 0–255 and a/b to 0–255 before undoing the scaling/offset, converts to BGR, and clips RGB to 0–255. The float entry divides by 255 and returns clipped float32; the uint8 entry **truncates** to uint8. Thus they share a continuous-domain formulation, but are not bitwise equivalent end-to-end at different input/output quantizations.

In `_stage_global`, contrast/brightness/tonal operations precede clarity and vibrance/saturation follow it. P6 supplies the decoded source directly to the isolated clarity operator, not the intermediate canvas of a complete recipe. Other stages may alter noise or clipping; this is deliberately not an end-to-end recipe comparison.

For low variance `v << ε`, `a≈0`, hence `base≈Q_r(Q_r(l))`. The residual is approximately `(Id−Q_r²)l`, including noise. At frequencies rejected by this base, the amplitude gain approaches `1+s`, with variance/energy gain `(1+s)²`. At recipe 4 these are approximately 1.04 and 1.0816; at 10, 1.10 and 1.21. This is a local linearized explanation, not a universal frequency response of the nonlinear guided filter. Strong edges have larger v and smaller relative removal into the residual, but edge preservation does not imply noise rejection in a subsequent residual boost. A perfectly constant input has essentially zero detail.

`sqrt(0.02)≈0.1414` in normalized L, equivalent to about 36.06 L255 DN or 14.14 L*. This is a variance regularizer, **not** an estimated noise standard deviation or an appropriate universal denoising threshold.

### Complete clarity implementation inventory

| Entry / consumers | Formulation and scale | Equivalent to global clarity? |
|---|---|---|
| `ColorGrader._add_clarity`, `_F_add_clarity`; recipe `_stage_global`, grader `_color_ops` | LAB L residual above; recipe /100, direct preset fractions | Same continuous formula; dtype/conversion differences remain |
| `ColorGrader.clarity_split`; engine finishing paths | `L − n(L−B_form) + p(L−B_texture)`; widths `max(int(.25 minHW),40)` and `max(int(.008 minHW),2)`, normalized ε=.05; optional final mask | No: two overlapping residuals, different scales and epsilon. Float input is 0–255; uint8 path still uses quantized LAB |
| `regions._local_clarity`; `apply_local_adjustment`, local-adjustment engine stage, Advanced Retouch | `L + s mask (L−B)`; float BGR/L255, width `max(1,int(.01 minHW))`, ε=50 in L255², normalized equivalent ~0.0007689 | No: different width, ~26× smaller normalized epsilon, mask, helper and sometimes coefficient resolution |
| `skin.local_clarity`; `perf_optimizations` face path | Per-BGR-channel Gaussian unsharp mask `I + s mask (I−G(I))`; actual Gaussian kernel `2 radius+1`; feathered nose/lips/eyes/bridge support | No: RGB rather than L, Gaussian rather than guided, masked face operation |
| `ColorGrader.add_impact_finish` | Wrapper calling global uint8 clarity at `.18 × clipped(strength/100)`, plus other tonal/saturation/glow operations and a final strength blend | Not a separate clarity kernel; the combined effect is not equivalent |

The face-path consumer supplies `skin.local_clarity` with radius `max(8,int(.04 face_width))|1` and strength `recipe_clarity/100 × .20` (recipe 4 → .008). Consequently a full face-aware recipe may apply both face-local and global clarity. **This experiment isolates global clarity, not that entire recipe.** Iris enhancement in [eyes.py](../../retouch/eyes.py) additionally contains a 3×3, sigma-1 Gaussian unsharp operation after color edits; it is independently controlled, not another implementation of the global clarity slider.

The region operator uses `utils.guided_filter`, not the grader's helper. It clips the denominator below at epsilon and, when the shorter image dimension exceeds 1200, computes coefficients on an area-downsampled canvas, upsamples them linearly, then averages them at full resolution. These helpers are not generally numerically identical. The utility docstring's claims that self-guidance is equivalent to bilateral filtering, that r means a 2r+1 kernel, and that increasing epsilon preserves more edges are not supported by its actual equations. No production docstring was changed.

The local-adjustment engine stage bridges float 0–1 to 0–255 but passes its supplied strength through; `apply_local_adjustment` expects fractional operation strength. The Advanced Retouch UI consumer performs its own /100 conversion. Do not infer local-operation units from the global recipe solely because both use the word clarity.

An entirely zero local-adjustment mask returns early. For a partly active region mask, unchanged L outside the mask still passes through the float color conversion; exact source-pixel identity there is not guaranteed by the region operator itself. The proposed bypass policy therefore requires explicit source restoration, not merely multiplying the L increment by zero.

## 3. Controlled experiment and evidence boundaries

Runner: [p6_clarity_experiment.py](../../scripts/qa/p6_clarity_experiment.py). Summary/viewer: [p6_clarity_diagnostics.py](../../scripts/qa/p6_clarity_diagnostics.py). Tests: [test_p6_clarity_experiment.py](../../tests/test_p6_clarity_experiment.py).

Final artifacts: [native review page](../../test_output/p6_clarity_20260907_final/review.html), [manifest](../../test_output/p6_clarity_20260907_final/manifest.json), [synthetic CSV](../../test_output/p6_clarity_20260907_final/synthetic.csv), [mixed-detail CSV](../../test_output/p6_clarity_20260907_final/mixed_detail.csv), [parameter CSV](../../test_output/p6_clarity_20260907_final/parameter_sensitivity.csv), [real ROI CSV](../../test_output/p6_clarity_20260907_final/real_rois.csv).

- 22 synthetic cases, 384×384, each generated with seeds 11/29/47: constant gray; clean smooth gradient; independent Gaussian residuals at 2 and 0.5 DN; correlated Gaussian residuals; Poisson-like noise generated in linear light then encoded; 4-DN-step quantized gradient; JPEG gradient quality 40 and 75; JPEG blurred edge; clean and noisy versions of hard edge, blurred edge, fine hair-like lines, sinusoidal fabric, printed checks and sparse pore-scale spots.
- Three seeds are noise realizations, **not three independent photographs**. Deterministic fixtures repeat across seeds. Hair, pores and fabric are controlled proxies, not authentic skin ground truth. Gaussian examples use achromatic encoded-RGB noise; Poisson-like generation is not a camera ISP simulation.
- Seven recipe values: 0/1/2/4/6/8/10. Nine arms give **4,158** synthetic rows. No candidate constants were selected using these results. Additional matched-noise structural probes give **756** rows. Sensitivity tests give **144** rows, using windows 5/15/58, epsilons .0008/.005/.02/.08, values 4/10 and six fixed fixtures.
- Within each comparison, source, crop, strength, base parameters and output encoding are held fixed. Parameter tests use the **same 116-pixel inset** across all windows. Standard synthetic metrics use a 12-pixel inset.
- Native source: `/Users/dennis/Desktop/arisaedited/DSCF1884.jpg`, 3879×5850; SHA-256 `e1aede4f43fb7e98951515eb527020510d3382a13a7f75dc7616eb46afa610cd`. Eleven reviewed ROIs × seven values × nine arms = **693** rows. This is one edited/rendered JPEG, one subject, no clean reference, no ISO/camera metadata establishing a sensor model. It is not a held-out photographic benchmark.
- Native crops are processed with width 58 and an additional 132-pixel context margin, then cropped to the fixed coordinates. All eleven current-arm crops at recipe 4 match a full native `_F_add_clarity` render with **max error 0 DN**. No face detector, candidate mark generator, learned model, skin repair or other retouch operation runs.
- Native ROI labels were corrected after source inspection: a nose crop is not a pure cheek patch, and a wig/background crop is not a bare face/background boundary. A separate cheek and face/wig crop were added. Algorithm constants were unchanged. Earlier diagnostic directories are superseded by `_final`, not deleted.

### Arms and reconstruction control

| Arm | Change relative to source |
|---|---|
| `disabled` | Exact input copy |
| `conversion_only` | Current float LAB round trip, no clarity gain; intentionally not an identity arm |
| `production_current` | Actual global LAB residual formula and direct LAB→BGR reconstruction; tested against live float implementation |
| `matched_current` | Same unfiltered residual, but delta-preserving reconstruction shared by all candidate gates |
| `soft_floor` | Boost only `sign(d) max(abs(d)−2σ,0)` |
| `variance_snr` | Boost `d g`, where `g=clip((V9(d)−1.5σ²)/(V9(d)+1e−12),0,1)` |
| `multiscale_snr` | `fine=l−G_sigma1(l)`, `middle=G_sigma1(l)−base`; independently variance-gated with σ and .30σ |
| `snr_edge_cap` | Variance-gated boost plus a 7×7 source-L min/max envelope when local range exceeds 16 L255 DN |
| `edge_coherence` | Variance gate additionally multiplied by clipped structure-tensor coherence/.5; intended stress test for oriented-detail preference |

Candidate σ is a **patch-wide** Haar-HH MAD estimate, floored at .25 L255 DN, not a calibrated local sensor-noise model. Constants 2, 1.5, .30 and the structural windows are predeclared research choices, not recommended production defaults. In particular, .30 is a heuristic band-noise factor, not a measured covariance propagation.

For candidates and matched control, reconstruction is `I + [RGB(L+increment,a,b)−RGB(L,a,b)]`, clipped to the legal RGB range, with exact zero-increment support restoration. This prevents conversion-only changes from being mistaken for a gate's effect or defeating bypass. **Candidate gains must be compared to matched_current as well as production_current.** This bridge is not established as a production-safe color/gamut solution.

### Metrics and what they mean

- Plane-detrended residual standard deviation; Gaussian sigma-2 high-pass RMS and local contrast; Haar-HH MAD; Hann-windowed PSD energy in .01–.05, .05–.15 and .15–.5 cycles/pixel. Metrics use floating display-referred luma DN, not another quantized LAB conversion. Detrended texture energy on a real ROI is a proxy, not identified sensor noise.
- Controlled corruption: RMS of `F(clean+corruption)−F(clean)`, divided by the disabled counterpart. This includes nonlinear response to corruption, so separate conversion/matched controls matter. Absolute error to clean is also recorded; a contrast enhancement can legitimately change clean structure, so clean-image error alone is not a quality score.
- Clean structural projection: correlate output high-pass with the known clean high-pass. This is **not enough** to establish preservation in a noisy image. Supplemental matched counterfactual `F(clean+noise)−F(flat+the_same_noise)` measures projected structure response **in noise**, with no clean guidance supplied to the candidate.
- JPEG/block diagnostic: excess squared first difference across aligned 8-pixel boundaries versus other locations. Crop origin is accounted for. This is not full PSNR-B; printed 8-pixel texture intentionally triggers it too. JPEG grid interpretation on previously resized or processed photos is uncertain.
- Export diagnostic: nearest-rounded uint8 PNG changed-pixel fraction, RMS delta and high-pass RMS, separate from prequantization metrics. This is not certification of every production export path, particularly uint8 truncation or subsequent JPEG re-encoding.
- Halos: signed profiles of midrange clean/blurred/JPEG edges; under/overshoot, reversed slopes, spatial extent and error area. [Plateau-relative halo CSV](../../test_output/p6_clarity_20260907_final/edge_halo_summary.csv) removes far-field conversion offsets before labeling localized excursions as halos. Raw absolute-to-clean values are retained separately. Noisy profiles can reverse slope simply because of noise; those counts are not proof of ringing.

The earlier Meitu comparison found that unregistered pairs invalidated an HF-based fabric/clarity inference. This study uses identical coordinates and clean synthetic truth; it does not reuse competitor renders as calibration truth. See [existing QA limitation](RESEARCH_MEITU_COMPETITOR_QA_2026_08_29.md).

## 4. Synthetic results

Three-seed mean; source 2-DN Gaussian measured RMS is 2.00009 DN. These are production-current results, including the current float conversion path.

| Recipe value | Strength | Paired noise RMS, DN | HP RMS ratio | High-band energy ratio | Native sky HP ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2.00009 | 1.00000 | 1.00000 | 1.00000 |
| 1 | .01 | 2.03350 | 1.01081 | 1.02240 | 1.00539 |
| 2 | .02 | 2.05255 | 1.02063 | 1.04243 | 1.01534 |
| 4 | .04 | 2.09065 | 1.04027 | 1.08309 | 1.03523 |
| 6 | .06 | 2.12878 | 1.05991 | 1.12452 | 1.05512 |
| 8 | .08 | 2.16693 | 1.07955 | 1.16674 | 1.07501 |
| 10 | .10 | 2.20509 | 1.09919 | 1.20973 | 1.09490 |

At recipe 4, HP RMS ratios for Poisson-like noise, correlated noise, JPEG Q40 gradient and quantized gradient are respectively **1.04146, 1.06057, 1.04589, 1.04369**. Thus the response is not exclusive to independent Gaussian noise. On the perfectly constant patch no spatial detail is created, although a uniform conversion shift can change rounded pixels. In particular, .5 encodes to 127.5 DN: a tiny shift across that rounding boundary can change 100% of pixels without creating grain. Changed-pixel fraction alone is not an artifact metric.

The 0.5-DN fixture gives a larger apparent paired-noise ratio (1.1632 at recipe 4), partly because the conversion floor is a larger fraction of this tiny signal. It must not be reported as a 16% intrinsic residual gain. The conversion-only control already changes low-amplitude statistics. Conversely, Haar MAD on the 2-DN fixture falls to **0.97036×** while HP RMS and PSD rise; on correlated noise it rises to **1.60358×**. A single MAD estimate does not characterize these transformed/quantized distributions reliably.

JPEG Q40 block excess rises from **1.41506 to 1.54579 DN²** at recipe 4; Q75 rises from .76770 to .84308. The matched bridge gives 1.51009 and .82332, still above source. Quantization steps also enter the detail signal even when this particular block metric is zero. The printed fixture has block excess ~650 DN² without JPEG corruption: it is a deliberate false-positive control for any block-only confidence rule.

### Candidate comparison, recipe 10

Corruption ratios are relative to disabled; structure columns are projected gain **in a 2-DN-noisy image**, not only on its clean twin. All values are three-seed means. A structure gain of 1 means preservation of the source response with no additional projected boost; it does **not** mean the pores were erased.

| Arm | IID noise RMS ratio | Correlated ratio | JPEG Q40 ratio | Hair gain in noise | Fabric gain in noise | Pore gain in noise | Printed gain in noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Disabled | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Conversion only | 1.00718 | 1.06480 | 1.07153 | .99994 | 1.00030 | 1.00222 | .99989 |
| Production current | 1.10250 | 1.14127 | 1.10770 | 1.08688 | 1.07982 | 1.08777 | 1.07385 |
| Matched current | 1.09565 | 1.07798 | 1.03551 | 1.08693 | 1.07953 | 1.08555 | 1.07396 |
| Soft floor | 1.00515 | 1.01842 | 1.01322 | 1.04682 | 1.02688 | 1.01408 | 1.04385 |
| Variance SNR | 1.00007 | 1.03619 | 1.02665 | 1.06761 | 1.05910 | 1.00157 | 1.06745 |
| Multiscale SNR | 1.00000 | 1.02567 | 1.02341 | 1.05845 | 1.04749 | 1.00216 | 1.06413 |
| SNR + edge cap | 1.00007 | 1.03619 | 1.02665 | 1.05423 | 1.05216 | 1.00151 | 1.06160 |
| Edge coherence | 1.00003 | 1.02126 | 1.02202 | 1.06761 | 1.04584 | 1.00072 | 1.01316 |

The IID result alone would misleadingly favor variance/multiscale gates. They largely abstain from boosting faint pores embedded in noise. On **clean** pore proxies the matched, variance, multiscale and orientation gains are 1.08161, 1.07177, 1.06203 and 1.01088; the mixed-detail test exposes the additional uncertainty caused by noise. Orientation coherence also suppresses legitimate isotropic/printed detail even when clean. It is not a universal structural-confidence gate.

### Guided-filter parameter sensitivity

On a fixed interior at recipe 4, reducing epsilon from .02 to .0008 with width 5 changes 2-DN paired noise RMS only from **2.09777 to 2.09232 DN**. Meanwhile clean hair-proxy gain changes from **1.03419 to 1.01764**, and the hard-edge gain from 1.01365 to 1.00182. It reduces desired enhancement along with some artifacts rather than separating noise.

With epsilon .02, changing width 5→58 changes Gaussian RMS from **2.09777 to 2.10081 DN**, increases hair gain from 1.03419 to 1.03865, and broadens edge influence. At width 58, increasing epsilon .0008→.08 raises hard-edge absolute overshoot from 0 to **1.515 DN**, with the usual conversion qualification. No tested global radius/epsilon adjustment is a noise classifier. Scaling width with whole image size also means downscaled previews and small faces cannot be assumed equivalent to native processing.

## 5. Halo, ringing and edge-detail tradeoffs

[Synthetic signed profiles](../../test_output/p6_clarity_20260907_final/halo_profiles.png) and [raw profile samples](../../test_output/p6_clarity_20260907_final/edge_profiles.csv) make the conversion offsets, local lobes and JPEG residuals separately visible.

| Clean hard edge, width 5 | Recipe 4 under / over, DN | Recipe 10 under / over, DN |
|---|---:|---:|
| Disabled | 0 / 0 | 0 / 0 |
| Current, relative to its own far plateaus | .581 / .677 | 1.451 / 1.693 |
| Matched current | .581 / .677 | 1.451 / 1.693 |
| SNR + local edge cap | .066 / .077 | .163 / .190 |

The current clean step develops dark and bright edge-adjacent lobes: a small halo/overshoot, not proof of a long oscillatory ringing train. For this width-5 step the plateau-corrected error beyond six pixels is zero, **not a general no-halo result**. The wider-window parameter tests show different spatial behavior. Blurred-edge current plateau-relative excursions are ~.036/.038 DN at recipe 4 and .092/.093 at 10. JPEG blurred-edge profiles already contain substantial oscillating error before clarity; the operator amplifies parts of it. Its far-edge error area grows from ~1.949 DN-pixels disabled to ~2.015 at recipe 4 and 2.137 at 10, after plateau correction. The gate does not remove the existing JPEG ringing.

The envelope cap reduces hard-edge excursions but is not a general halo cure. Its 7-pixel envelope can miss wider lobes, and on clean printed checks it reduces projected gain to **.99970** at recipe 10 versus matched 1.07393. It also reduces clean hair boost from 1.08696 to 1.05354. A claim that it preserves all useful microcontrast would be false. Strong local range is not the same as an unwanted boundary.

Native [boundary-change plots](../../test_output/p6_clarity_20260907_final/real_boundary_deltas.png) and [B/G/R profile samples](../../test_output/p6_clarity_20260907_final/real_edge_profiles.csv) cover hair/sky, wig/background, face/wig and costume/background transitions. These are five-row native profiles of output minus source, not clean-reference halo measurements. They expose edge brightening/darkening but cannot tell whether a pre-existing halo belongs to the capture, prior editing or this operation. An unobscured bare-face/background boundary is **not available in this portrait**, whose silhouette is largely wig; a reviewed real example remains required. The synthetic hard/blurred steps are controlled boundary surrogates, not a substitute for that missing scene.

## 6. DSCF1884 native evidence

All coordinates use original pixels `(x0,y0,x1,y1)` and are recorded in the manifest. Owner review should use 100% native crops, not contact-sheet appearance.

| ROI | Coordinates | Source HP RMS DN | Current recipe-4 HP ratio | Matched-current ratio | Soft-floor ratio | Variance ratio | Multiscale ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Sky | 150,120,950,620 | .5817 | 1.03523 | 1.03979 | 1.01530 | 1.02779 | 1.02208 |
| Soft background | 2950,2650,3350,3050 | 1.3977 | 1.03483 | 1.03885 | 1.02123 | 1.02554 | 1.01833 |
| Cheek with makeup | 1790,1780,1890,1890 | 2.1124 | 1.03193 | 1.03561 | 1.02053 | 1.01873 | 1.01386 |
| Wig/hair | 1300,1220,1680,1520 | 3.0750 | 1.03226 | 1.03324 | 1.02970 | 1.03201 | 1.02851 |
| Eyes/lashes/brows/makeup | 1420,1560,1930,1770 | 4.6124 | 1.02607 | 1.02504 | 1.02013 | 1.02215 | 1.01855 |
| Costume fabric | 680,3270,1080,3710 | 3.7177 | 1.03410 | 1.03409 | 1.02749 | 1.03102 | 1.02542 |

The sky's float round-trip-only RMS change is **.36037 DN**, while the isolated matched clarity increment at recipe 4 is only **.02595 DN RMS**. Current sky HP increases from .5817 to approximately .6022 DN; the change is small at native view. Current recipe 4 changes **1.27%** of sky pixels after the harness's nearest-rounded PNG export; recipe 10 changes **18.10%**. The matched and gated sky arms remain identical to the source after that particular 8-bit rounding at both strengths, despite nonzero float changes. Thus the experiment must not conflate float noise energy, rounding sensitivity, and clearly visible grain.

The important estimator failure: sky Haar-HH σ is only **.01154 L255 DN**, so the candidate uses its .25-DN floor. Variance gating retains about **70% of matched-current sky HP gain**, versus about 96% of hair gain. It is reacting to correlated/processed residuals as signal. Its source-exact bypass fraction is only .033% on sky, but **44.4% on the cheek**. Soft thresholding bypasses ~61.5% of sky pixels but also loses more projected structure under synthetic noise. “Low noise estimate” is not evidence of reliable high-SNR detail.

Native source/current/candidate crops for sky, wig strands, eye makeup, fabric and the reviewed boundaries were inspected. Recipe-4 differences are subtle; the catastrophic historical output is not reproduced. Strands and lash/makeup boundaries remain visible in the inspected variants, but one JPEG cannot prove authentic pore retention, distinguish eyelashes from drawn lines, or certify that skin grain is useful texture. This report makes no winner claim from the real ROI energy ratios.

## 7. Primary literature that changes the recommendation

Search stopped after these sources covered decomposition, noise modeling, low-amplitude gain limiting and artifact measurement; no beauty/diffusion or learned full-face retouch survey was undertaken.

| Primary source | Direct relevance and limitation |
|---|---|
| [He, Sun, Tang, Guided Image Filtering, ECCV 2010](https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf) | Local linear coefficients, epsilon regularization and base/detail enhancement explain the existing operator. Fast edge-aware filtering does not make its residual a noise-free structural signal. The paper's demonstrations are not Retouch parameter validation. |
| [Paris, Hasinoff, Kautz, Local Laplacian Filters, SIGGRAPH 2011](https://people.csail.mit.edu/sparis/publi/2011/siggraph/Paris_11_Local_Laplacian_Filters_lowres.pdf) | Section 5.2 explicitly discusses noise/compression amplification and blends small-amplitude remapping toward identity. Its example transitions at 1–2% of maximum intensity motivate conservative gain taper, not fixed Retouch thresholds. Edge-aware multiscale remapping is a plausible later challenger; the P6 multiscale arm is **not** this paper's algorithm. |
| [Farbman et al., Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation, SIGGRAPH 2008](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/08/Farbman-EPD-small-SG08.pdf) | Weighted least-squares decomposition separates structural scales and addresses limitations of repeated bilateral smoothing. It motivates scale-specific policies, but a better base alone does not identify sensor/JPEG noise; solver cost and Retouch-specific edge tests would remain. |
| [Pham and Jeon, Efficient image sharpening and denoising using adaptive guided image filtering, 2015](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2013.0563) | Direct sharpening/denoising work using adaptive guided parameters and MMSE reasoning. A bounded alternative if the current estimator remains inadequate; not reproduced here and no transfer of its claimed artifact performance to Retouch. |
| [Foi et al., Practical Poissonian-Gaussian noise modeling and fitting; author resources](https://webpages.tuni.fi/foi/sensornoise.html) | Signal-dependent noise motivates a calibrated variance model rather than one image-wide sigma. Raw noise assumptions do not directly identify noise in an edited, gamma-encoded JPEG with unknown denoising/demosaicing history. |
| [Yim and Bovik, Quality Assessment of Deblocked Images, 2011](https://live.ece.utexas.edu/publications/2011/cy_tip_jan11.pdf) | Supports measuring blocking explicitly instead of relying only on global error/energy. The simple P6 aligned-boundary statistic is inspired by this problem setting, not a full PSNR-B reproduction or a semantic texture classifier. |

Bilateral/guided substitutions and full local-Laplacian/WLS methods remain literature-only challengers. The executable sensitivity and multiscale experiments test narrower hypotheses. No unimplemented paper was assigned a measured score.

## 8. Ranked next steps, abstention and falsifiable production gates

1. **RESEARCH_ONLY: confidence-qualified local variance/SNR gain, with explicit bypass and soft-floor control.** It best preserves clean hair/fabric boost among simple tested gates while almost avoiding IID noise amplification. It still fails on real processed sky and barely boosts noisy pore proxies. The next work is confidence calibration, not tuning 1.5 until this sky looks good.
2. **RESEARCH_ONLY: multiscale noise-aware gain with calibrated per-band noise propagation.** Better correlated-residual suppression in this pilot, but more detail-boost loss and no edge-halo advantage. The .30 band factor must not enter production on this evidence.
3. **RESEARCH_ONLY: soft-floor attenuation as a cheap baseline.** Useful on real smooth sky and correlated residuals; more conservative on legitimate hair/fabric in noise. It adds no clarity when evidence is faint; that can be correct abstention, but is not proof of structural recovery.
4. **NO-GO as global replacements: orientation-only confidence and the tested local min/max edge cap.** Orientation loses isotropic detail; the cap suppresses legitimate printed/hair enhancement and does not guarantee wide-halo control. They could become narrowly supported auxiliary mechanisms after separate validation.
5. **RESEARCH_ONLY, lower priority: adaptive guided/WLS/local-Laplacian challenger.** Only worthwhile if improved noise confidence plus bounded gains cannot satisfy the next controlled test. No production rearchitecture is justified by this pilot.

Proposed confidence policy, **implemented as an opt-in mechanism but not numerically calibrated as a universal policy**:

- Estimate band noise only from trusted smooth neighborhoods; account for luminance dependence, correlation, JPEG/block evidence and estimator disagreement. A nearly zero HH estimate in a visibly varying smooth patch is a reason to distrust the estimate, not permission to boost.
- If structural energy is indistinguishable from the uncertainty interval of noise energy, set the **additional gain to zero**. Do not shrink the original photographed detail as part of clarity. Likewise bypass externally reviewed smooth supports; abstain across uncertain semantic/mask boundaries instead of filling them with a hard gate.
- If a noise-gain budget is adopted, use an upper uncertainty bound: in a noise-dominated band, amplitude gain is approximately `1+s g`. A proposed 1% RMS budget would require `s g≤.01`, but this is an engineering target to validate, not an established human-visibility threshold. With no reliable bound, bypass rather than asserting safety.
- Avoid abrupt spatial gain changes: confidence/support transitions themselves need halo/seam tests. The current patch-wide estimator is not a production tile estimator; overlapping tiles and identical-context behavior remain untested.
- If the frame is too small, soft, noisy or processed to support a stable estimate, global bypass is valid. No blanket ISO cutoff is inferred from a JPEG lacking ISO metadata.

### Next validation experiment (outside this production patch)

Keep the current decomposition and matched reconstruction fixed. Compare **disabled / current / variance gate / soft floor / multiscale gate**, first with an independently known synthetic noise covariance and externally reviewed smooth supports, then with an estimated noise model. Separate the oracle-versus-estimated gap from the representation/gain gap. Do not use a detector, face parser or a learned retoucher to hide that gap.

Minimum useful next photographic set: at least **six owner-mapped images from three distinct capture sessions/subjects**, one calibration image and one locked evaluation image per session; report that this is session-balanced rather than subject-held-out. Include low/high-noise sources, a genuinely exposed face/background edge, hair against sky, patterned fabric, skin with resolvable fine detail, and a small/soft-focus case. Prefer registered low/high-ISO or repeated-frame pairs plus the original RAW and stated development settings for at least one controlled scene. New derivatives must retain native dimensions; synthetic JPEG/noise controls should be made from the clean reference, not existing retouched renders. A true subject-held-out study needs additional subjects, not relabeling crops from DSCF1884.

Predeclare a sweep of faint-to-strong hair/pore/cloth contrast against multiple noise amplitudes and correlations, including unknown JPEG phase/resizing, before collecting evaluation metrics. Lock constants on a separate development seed/capture set. Evaluate output increments and true structural projection, paired corruption, native owner review, broad luminance/chroma drift, and edge profiles together.

Suggested **future acceptance targets**, not claims achieved by this run: upper-bound smooth-region noise RMS gain ≤1% and high-band energy gain ≤3% on eligible supports; retain ≥90% of current additional structural boost on independently judged high-SNR hair/fabric/eyes; abstain rather than demand pore boosting at unidentifiable SNR; no new >1-DN plateau-relative halo and no significant extension of existing lobes at requested strengths. Also measure gate coverage/abstention, mask seams, output rounding, resizing and all affected clarity entry paths. Targets must be agreed with the owner and tested independently, not optimized on the current photograph.

These findings would be falsified or materially changed by a calibrated estimator/gain pair meeting those locked noise/detail/halo limits on native held-out captures with owner approval. Conversely, failure even with oracle noise and trusted supports would demote gain gating and justify a decomposition challenger. Evidence that apparent noise is deliberate texture would favor local bypass/user control over automatic noise classification.

## 9. Verification, reproducibility and scope

Commands (run from repository root; choose a new output directory on repetition because overwrites are refused):

```bash
PYTHONDONTWRITEBYTECODE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python -m pytest -q tests/test_p6_clarity_experiment.py -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python scripts/qa/p6_clarity_experiment.py test_output/p6_clarity_20260907_final --photo /Users/dennis/Desktop/arisaedited/DSCF1884.jpg
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qa/p6_clarity_diagnostics.py test_output/p6_clarity_20260907_final
PYTHONDONTWRITEBYTECODE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python -m pytest -q tests/test_p6_clarity_experiment.py tests/test_float_pipeline.py::TestFAddClarity -p no:cacheprovider
```

Harness tests: **25 passed**. Coverage includes exact current float-operator parity at all seven strengths; zero identity and finite/range checks for all arms; no detail created on constants; deterministic fixture coverage; controlled Gaussian response; valid fixed-interior halo metrics; padded/full-context equality; plane detrending; JPEG-grid origin accounting; exact counterfactual identity; zero-signal gating; refusal to overwrite runs and rejection of invalid metric interiors. These are focused P6 tests, not a full Retouch release test suite.

The pre-implementation combined harness command passed **29 tests in 0.45 seconds** (25 P6 plus four existing float-clarity tests; no full engine/model execution). The final experiment took **59.697 seconds**, with NumPy 1.26.4 and OpenCV 4.11.0. Diagnostics generated successfully; the sandbox emitted font-cache-directory warnings, but the rendered charts were inspected. The final directory contains 363 PNGs and approximately 65.35 MB of evidence in total.

Post-implementation validation previously passed **561 focused tests in 12.92 seconds**: the six original opt-in clarity tests, 25 P6 harness tests, global clarity dispatch, float-clarity, grading internals, parameter/recipe integration and the engine suite. The updated opt-in suite now has seven tests and adds invalid-input/dtype rejection; its current combined P6 run is recorded by the productionization handoff. No full model-backed release suite was run. The production files compile with `py_compile`; `git diff --check` is clean. The opt-in native smoke measurements are recorded in the implementation follow-through section above.

The pre-implementation audit checked all CSV numeric fields for finiteness, expected row counts, eleven zero-DN native crop parity results, script syntax, report/viewer links, and runner/production hashes. Its clean-diff assertions describe that earlier research snapshot only. After the scoped implementation, `git diff --check` remains clean and the intended P6 production/API edits are present and unstaged. P4/P5 follow-through files are tracked separately in the current working tree and are not part of the P6 evidence claims.

The final manifest records source and production-file hashes, versions, constants, counts and elapsed wall time. The offline implementation caches several candidate fields and is not a production runtime/memory benchmark; candidates require extra linear-size maps and local filters over the current O(N) base. No per-arm peak-memory claim is made.

Only the explicitly scoped P6 production files (`retouch/grading.py`, `retouch/engine.py`) and the API documentation were changed to expose the opt-in gate in the P6 follow-through; legacy production defaults and recipes remain untouched. Separate P4/P5 productionization edits are reviewed outside this report. No source photograph was written, and no staging or commit was performed. The two P6 QA scripts, isolated P6 tests and this report remain research artifacts; generated evidence lives under ignored `test_output/`. The `_final` manifest's source/production hashes describe the pre-opt-in research run; the opt-in code does not alter its legacy arm, so those measurements remain valid for the default path, but the manifest is not a hash attestation for the newer opt-in implementation.

### Owner-review shortlist

1. Native sky and soft background at recipe 4, then 10: input/current/soft-floor/variance/multiscale; distinguish visible grain from sub-DN numerical changes.
2. Wig strands, eye/lash makeup and costume fabric: does the reduced added contrast remain useful? Do not label makeup edges as sensor noise.
3. Hair/sky, face/wig and costume boundaries with signed delta maps and profiles; distinguish source halos from newly added lobes.
4. Clean versus noisy pore-proxy and printed-texture diagnostics: confirm that withholding uncertain extra boost is acceptable rather than choosing a gate solely for lower noise energy.

**Final P6 disposition: reproducible small intrinsic noise/codec-residual amplification; no recurrence of the historical severe artifact demonstrated; RESEARCH_ONLY for candidate gain gating and NO-GO for an automatic production replacement until independent confidence, detail and halo evidence exists.**
