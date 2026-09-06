# FA-02: first offline representation harness

Scope: implement the first experiment from [the completed representation research](RESEARCH_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md), without production wiring, learned models, classification, candidate generation, or changes to the smoother. This document records the runnable contract and validation boundary; it is not another literature review.

## Evidence boundary and corpus

The available project artifacts did not supply an owner-confirmed subject/session split **and** externally accepted native-resolution restoration supports/texture annotations. No real portraits were enrolled. In particular, previously inspected Meitu anchors were not relabelled as untouched holdout subjects. Their old 2048-pixel renders / 1600-pixel analyses are not native pore ground truth.

The executed corpus is **11 mathematical signal canvases, zero portraits, zero people, zero held-out subjects**. Ten canvases are 512×512; one is 128×128. These are harness-validation fixtures, not the proposed 12-portrait / 48-patch benchmark or a real-image quality pilot. There is no automatic winner.

| Fixture | Precisely controlled difference |
|---|---|
| `clean_signal` | Analytic Gaussian dots, thin curved lines, flat regions, and a retained rectangular edge; S is a fixed sigma-2 Gaussian of X |
| `constant_tone` | BGR offset [10,8,6], without useful detail |
| `broad_ramp` | A linear luminance ramp, without useful detail |
| `corrected_spot` | A known −40-level disk, radius 7, present only in X; exact correction support supplied |
| `held_makeup_edge` | A known dark stroke present only in X; external protected rectangle supplied |
| `noise_seed_1701`, `noise_seed_2909` | Independent Gaussian noise, sigma 2.5 encoded levels, added to the same clean X; identical clean S |
| `jpeg_phase_0_0`, `jpeg_phase_3_5` | OpenCV JPEG quality 30 at two declared block phases; identical clean S; includes uint8 rounding/chroma subsampling |
| `soft_focus` | Sigma-1.5 source blur followed by fixed sigma-2 smoothing |
| `small_sampling` | Analytic host downsampled 4× to 128×128; sigma-0.6 smoothing, no upsampling |

The injection equations supply supports, not a detector. All six arms use exactly the same X, S, masks, crop coordinates, nominal face scale, and output encoding within a case. Noise/JPEG/spot/stroke cases also share S and supports with `clean_signal`, permitting causal differences between restored increments. These controls do **not** simulate verified high-ISO faces, biological pore distributions, cosplay makeup, or real repair behavior.

## Added files and invocation

- `scripts/qa/fa02_texture_representation_experiment.py`: manifest validation, transforms, support propagation, reconstruction, subprocess measurements, diagnostics, and review index.
- `scripts/qa/fa02_texture_controls.py`: deterministic mathematical fixtures only.
- `tests/test_fa02_texture_representation_experiment.py`: focused mathematical, safety, provenance and end-to-end checks.
- This report. Existing production modules, prior research, FA-03 artifacts and corpus tooling are not edited.

From the repository root, choose **new** output paths (existing locks/runs/fixtures are never overwritten):

```sh
.venv/bin/python scripts/qa/fa02_texture_representation_experiment.py controls test_output/fa02_texture_representation_20260906/controls
.venv/bin/python scripts/qa/fa02_texture_representation_experiment.py freeze test_output/fa02_texture_representation_20260906/config_lock.json
.venv/bin/python scripts/qa/fa02_texture_representation_experiment.py run test_output/fa02_texture_representation_20260906/controls/manifest.json test_output/fa02_texture_representation_20260906/config_lock.json test_output/fa02_texture_representation_20260906/run --repeats 5
.venv/bin/python -m pytest -q tests/test_fa02_texture_representation_experiment.py
.venv/bin/python scripts/qa/fa02_texture_representation_experiment.py profile test_output/fa02_texture_representation_20260906/controls/manifest.json test_output/fa02_texture_representation_20260906/config_lock.json test_output/fa02_texture_representation_20260906/timings --repeats 10
```

Run tests separately from performance measurement. A completed run has `COMPLETE.json`; a partial directory is not evidence of success. Float arrays are authoritative. Native PNG review copies are rounded to uint8 and may hide sub-level changes. The signed diagnostic maps use fixed ranges across arms, never independent automatic contrast stretching.

## Frozen arms

Let D=X−S. All arms reconstruct `O=clip(S + 0.10 M detail, 0,255)` in encoded BGR float32. No grain, source borrowing, extra sharpening, noise-reliability weighting, chroma switch, resampling, or energy normalization is applied.

At nominal inter-eye distance 200, the full-resolution Gaussian stack has sigmas `[.6,1.2,2.4,4.8,9.6]`, explicit radius `ceil(3 sigma)`, and `REFLECT_101` padding. Scale follows supplied inter-eye distance, with a .6-pixel floor; collapsed bands are logged, not upsampled into invented detail.

| Arm | Detail signal |
|---|---|
| A0 disabled | 0 |
| A1 raw | D; the existing restoration algebra under the same external M, not the production dimensional-mask policy |
| A2 DoG | G.6(D)−G1.2(D) |
| A3 selectable multiscale | `[G.6−G1.2](D) + .5 [G1.2−G2.4](D)`; remaining band weights zero |
| A4 orientation-aware | A3 with a bounded source-structure-tensor attenuation on each band, retaining an isotropic path |
| A5 existing frequency | `.5 mid(D)+high(D)` from the actual `FrequencySeparator.separate`; low gain zero |

A3 exposes all adjacent band weights through the frozen config; one-hot equivalence and equal-weight telescoping are tested. It is an undecimated Gaussian/Laplacian band-difference stack, **not** a decimated `pyrDown`/`pyrUp` reconstruction.

A4 uses the dominant source-gradient normal, eigenvalue coherence, source-gradient strength and squared alignment of each band gradient to that normal. `gate=1−.5*coherence*strength*alignment²`, with `strength=sqrt(Jxx+Jyy)/(sqrt(Jxx+Jyy)+10)`. Tensor smoothing is sigma 1.2 at reference scale. The gate remains between .5 and 1, including an isotropic path; no alternate smoother is introduced. It can suppress genuine hairs as well as edges and is explicitly a challenger, not a proven improvement. Its impulse response is stimulus-dependent, not a linear MTF.

A5 uses face width 500 in the 512-pixel controls: the existing adaptive finite kernels are 61 and 21 pixels. Applying the linear separator to D is checked against subtracting separately extracted X/S coefficients. No call to `combine()` is made.

The initial settings came from the research design and an explicitly recorded a-priori validation declaration; they were not fitted to the fixtures or selected from the reported results. The config hash is frozen before the controlled run. This is **not** a development-selected production setting.

## External supports and leakage isolation

Each case supplies `allow`, `corrected` and `protected`, normalized H×W masks. Corrected/protected supports are disjoint and both forbid changes. No source hole is cut before filtering.

All arms share a conservative square footprint guard for the largest specified Gaussian band, the band-gradient support, tensor support, and the existing frequency kernels, including currently unselected stack bands. At reference scale its radius is 30 px. Chebyshev distance matches the square finite kernels. A four-pixel ramp begins **outside** that radius; the output edge is also guarded. This sacrifices coverage equally for every arm and avoids rewarding a larger mask only in one arm. Arbitrary soft `allow` boundaries remain externally supplied and can themselves modulate the output spectrum.

The report separates pre-gate coefficient contamination from post-gate changes. Zero post-gate correction leakage in these controls validates the common guard; it does not establish that DoG intrinsically recognizes or removes defects. No guard-size ablation or mask optimization is mixed into this comparison.

## Diagnostics and measurement interpretation

Each arm/case emits:

- `signed_arrays.npz`: float output, actual delta, requested delta, signed pre-gate detail, selected bands, clipping map, eligibility, and A4 fields.
- `output_native.png`, `recovered_signed_detail.png`, `pre_gate_signed_detail.png`; coarse luminance/chroma maps; signed selected-band maps.
- Fixed-coordinate native `X | S | O` ROI strips, signed local PSD arrays/images, source/S/O profile plots and raw profile samples.
- Corrected/protected Chebyshev ring min/mean/max/RMS, pre-gate coefficient RMS, halo CSV/SVG, forbidden-pixel maxima and signed contrast recovery with undefined-denominator handling.
- Paired-clean nuisance deltas and pre-gate maps, nuisance projection and ring statistics; independent-noise-draw differences and JPEG block-phase discontinuity proxies.
- Per-ROI mean luminance, coarse luminance RMS, chroma shift in declared B−Y / R−Y coordinates, and a flat variance-change **proxy**.

Supplied pore locations are measured for signed contrast recovery, local-minimum displacement and added local minima inside their supplied neighborhoods. Hair cross-sections supply contrast, position, width, extra minima and sparse-trace continuity counts. These measurements do not find/label new facial marks. Without reviewed, observable real features they are mathematical dot/line metrics only. Edge profiles record gradient width/location and overshoot/undershoot; raw profiles and native owner review remain necessary for ringing interpretation.

`transform_checks/` contains finite impulse arrays, amplitude spectra and CSV response samples. The finest residual and coarse base are also saved as **diagnostics only**, not silently restored. Low-pass output diagnostics deliberately include spectrum introduced by spatial compositing; transform DC rejection does not imply that arbitrary masked output has exactly zero local mean.

Each case/arm runs in a fresh process with one OpenCV thread. One first-call and five warm extraction-plus-reconstruction timings are recorded. Shared support-map time is separate. Peak RSS is sampled from the process high-water mark **before** metrics, FFTs, compression and rendering; it includes interpreter, imports, inputs, supports and compute. Incremental RSS is the rise above pre-compute high-water, not exact live transform allocation. Small fixtures can show zero increment. This is CPU microbenchmark evidence on these canvas sizes, not engine latency or a 4-megapixel/native-face scalability certification.

## Supplying a real development pilot later

The runner consumes saved arrays; it never enters the image pipeline to obtain them. A portrait manifest uses `purpose=development_pilot` and `split=dev` or `calibration`. Each case must supply:

1. Stable `person_ids`, `session_id`, `owner_mapping_reference`, and whether previously inspected. The same person/session cannot cross splits.
2. `native_source_reference` plus original SHA-256, `resampled=false`, profile/encoding, fixed crop `[x,y,width,height]`, inter-eye/face width in native pixels, and `smoothing_provenance` for the actual shared X/S capture.
3. An NPZ `arrays` path and SHA-256 with matching finite float32 BGR X/S in [0,255] and all three external masks in [0,1]. `support_status=externally_accepted` and an auditable `support_acceptance_reference` are required. These are assertions with provenance, not automatic proof of human acceptance.
4. Reviewed ROI rectangles and tags, optional visible pore centers/radii, hair/edge cross-sections, and explicit observability gaps. Do not annotate an unresolved source as a recoverable pore. Coordinates are local to the saved crop; the original crop offset is retained separately.

The generated control manifest demonstrates exact field shapes, but is not a substitute for owner mapping. For an untouched final comparison, use `purpose=locked_comparison`, `split=locked_test`, an owner holdout acceptance reference and the **accepted config hash**. A lock must declare `development_selected` and point to its development evidence. Previously inspected anchors are rejected from locked_test. These structural checks cannot prove an owner has never seen a subject; human provenance remains required.

The smallest useful next data delivery is two reliably mapped **development** subjects with saved native X/S and reviewed texture, flat/noisy, edge and correction patches. That can establish a real-image feasibility pilot, not a held-out winner. The planned approximately 12 portraits / 48 reviewed patches, with subject-separated development/holdout and genuinely small, soft, noisy, profile and difficult-light cases, is still missing.

## Decision boundary

DoG remains the simplest **research-prior candidate to test next**, not a validated production winner. A valid comparison must show useful photographed pore/hair recovery beyond disabled at a tolerable nuisance/halo/coverage cost, on native reviewed data. Fixed equal gain alone is not a matched-recovery comparison. The harness intentionally stops short of production integration.

Missing evidence: owner mapping and untouched holdout; externally accepted native supports and X/S; observable real pore/hair labels; representative noisy/JPEG/makeup/lighting cases; development-only band/gain selection; native owner preference/safety review; and larger-canvas/concurrent-worker performance. No change to `restore_micro_texture()` or defaults is authorized by this run.

## Executed results

**Completed: 11 cases × 6 arms = 66 outputs; 46 focused tests passed.** No full project suite or engine integration test was run. Native artifacts live under `test_output/fa02_texture_representation_20260906/` (generated/ignored by git, approximately 812 MiB including float diagnostics); source code and this report are separately reviewable.

Config SHA-256: `5a2696df1b3ab498cf850b8885548e06b6f5c769216a9e9f33d28e5ca62e11fe`.

The following is a **mechanism check**, not a portrait leaderboard. Dot/line columns are the recovered fraction of contrast lost from X to S, averaged over the 16 supplied dots / nine line cross-sections in `clean_signal`. Noise/JPEG columns measure the RMS of the arm's incremental output **minus its paired clean increment**, in encoded 0–255 levels, over common eligible pixels. Noise uses seed 1701; JPEG uses phase (0,0). Raw's approximately 10% contrast recovery is expected from the fixed .10 gain. Lower nuisance values can coincide with less useful signal recovery.

| Arm | Lost dot contrast recovered | Lost line contrast recovered | Returned noise RMS | Returned JPEG RMS | Constant-offset coarse luma RMS |
|---|---:|---:|---:|---:|---:|
| Disabled | 0% | 0% | 0 | 0 | 0 |
| Raw residual | 10.00% | 10.00% | 0.2462 | 0.1577 | 0.7223 |
| One DoG | 4.18% | 3.62% | 0.0861 | 0.0164 | 0 |
| Selectable multiscale | 5.50% | 4.70% | 0.0981 | 0.0208 | 0 |
| Orientation-aware multiscale | 5.48% | 4.44% | 0.0969 | 0.0187 | 0 |
| Existing frequency bands | 9.96% | 9.73% | 0.2447 | 0.0502 | 0 |

All arms have the same **70.77% eligible pixel coverage** at 512×512 under the deliberately conservative guard. All 66 outputs have exactly zero change in externally forbidden pixels. Paired corrected-spot and held-makeup-stroke output differences are zero for all arms. This is the guard's success, not a representation-quality win.

The correction **does** enter neighboring coefficients before gating: maximum RMS among measured external spot rings is 2.574 for DoG, 3.700 for multiscale, 3.064 for the orientation variant, and 6.800 for the existing frequency bands. Post-gate paired ring differences are zero. The runner saves both sides of that distinction.

The band-limited arms reject the constant and linear-ramp controls at the scored interior (zero after float32 reconstruction in this run). This does not promise zero local tone change around arbitrary mask/edge modulation. For example, the clean DoG output's coarse-luma RMS is `2.28e-5`; orientation gating produces `1.01e-4` despite excluding the coarse base. Existing frequency separation rejects broad DC but its high-band inclusion returns almost as much added white noise as raw residual under this setting.

On the 128×128 control, the .6-pixel sigma floor collapses the first two bands. A2/A3/A4 consequently equal disabled. This is explicitly logged, not treated as a successful small-face restoration or used to claim lower noise. On the soft-focus control, DoG's output-detail RMS is 0.00434 versus raw's 0.02183; without real visible features, neither number establishes better texture quality.

### Native owner-review shortlist

Open `run/native_review.html` without fit-to-page resizing, or inspect these exact artifact groups:

1. `run/clean_signal/A{1,2,3,4}_*/dot_signal_X_S_O_native.png` and `line_signal_X_S_O_native.png`: the useful-signal versus attenuation tradeoff; inspect float profiles for changes below one 8-bit level.
2. `run/noise_seed_1701/` and both `run/jpeg_phase_*/` arms: `paired_nuisance_signed.png`, flat/edge PSD and retained-edge profiles. These separate returned nuisance from the clean signal, including block-origin dependence.
3. `run/corrected_spot/` and `run/held_makeup_edge/`: `paired_nuisance_pre_gate.png` versus `paired_nuisance_signed.png`, halo rings/profiles and common eligibility. Confirm that coefficient spread is not misreported as visible leakage.
4. `run/small_sampling/` and `run/soft_focus/`: inspect collapsed-band/no-recovery behavior before considering a real small/soft-face prototype.

There is no real-portrait owner-review shortlist yet because those cases lack the required mapping/support package. The five previously inspected source anchors remain possible **development** sources only after owner mapping and native capture/annotation; the old Meitu outputs are optional appearance references, not pore targets.

The artifact-generating run briefly overlapped a focused test execution. Its embedded timing measurements are therefore **not the reported performance comparison**. A separate serial `profile` run, after the tests, records ten warm repetitions per arm/case without diagnostic rendering or concurrent tests. Its `timings/TIMINGS.json` is the timing authority; it records its own runner/config/manifest hashes. The later CLI profiling addition does not change any arm's transform, masks or reconstruction settings.

### Measured cost

Environment: macOS 26.6.2 arm64, Python 3.9.6, NumPy 1.26.4, OpenCV 4.11.0, one OpenCV thread. Table covers the **ten 512×512 cases only**; the 128×128 case is separately present in the timing JSON. Warm time is the median of the ten per-case medians. P95 pools their 100 warm samples and is descriptive, not a confidence bound. Peak RSS / incremental high-water are the maximum observed among those ten fresh case processes.

| Arm | First-call median ms | Warm median ms | Warm pooled p95 ms | Peak RSS MiB | Max incremental high-water MiB |
|---|---:|---:|---:|---:|---:|
| Disabled | 3.58 | 3.44 | 6.79 | 92.92 | 13.13 |
| Raw residual | 5.86 | 4.46 | 7.14 | 92.73 | 12.42 |
| One DoG | 8.54 | 5.78 | 11.20 | 96.19 | 16.50 |
| Selectable multiscale | 9.37 | 7.63 | 14.66 | 102.36 | 21.77 |
| Orientation-aware multiscale | 23.32 | 13.79 | 27.71 | 122.59 | 45.89 |
| Existing frequency bands | 18.94 | 10.50 | 24.24 | 99.61 | 18.91 |

The disabled harness arm still allocates/composites diagnostic-compatible float buffers; its cost is **not** the cost of bypassing restoration in production. Peak RSS includes loaded inputs and the Python/OpenCV process, not just added texture buffers. Do not extrapolate these figures into whole-engine speedups. A4's extra tensor/gradient work has not earned its added cost through verified real-hair or edge-safety evidence.

Visual verification was limited to native dot/line strips and fixed-range noise / pre-gate / post-gate correction maps; these rendered at their expected pixel dimensions and showed the expected guard regions. It was not exhaustive review of all 66 outputs, a real-face review, or owner acceptance. The very weak analytic increments are often below one 8-bit level, reinforcing the need for the saved float profiles.
