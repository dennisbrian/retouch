# P5 — Analytical self-blend tone operators

Requested 2026-09-06; completion review 2026-09-07 (Asia/Kuala_Lumpur).
Original scope: research, deterministic fixtures and offline tooling only. The
2026-09-07 production follow-through adds an isolated analytical module,
unit tests, and explicit caller-only grading/Engine routing, but still makes
no recipe/UI change, P4 change or photographic look recommendation.

## Decision

**GO for analytical operator infrastructure, an isolated opt-in module,
deterministic fixtures, offline verification, domain experiments and future
golden comparison tooling.**

**NO-GO for Photoshop pixel-parity claims or automatic recipe/pipeline
integration.** No actual Photoshop duplicate-layer exports were available. The
capture script has a DOM-reference audit and a mocked control-flow test, not a
Photoshop runtime test. The analytical module and caller-only Engine route are
unit-tested, but this is not a claim that existing RetouchEngine recipes or
Photoshop are pixel identical.

| Mode | MATHEMATICALLY DERIVED | SPECIFICATION VERIFIED | RETOUCH ENGINE VERIFIED | PHOTOSHOP VERIFIED |
|---|---|---|---|---|
| Multiply | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Screen | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Overlay | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Hard Light | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Soft Light | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Color Dodge | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Color Burn | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Exclusion | GO | GO — W3C definition checked | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Linear Light | GO for stated bounded operator | Qualified GO — canonical definition, not W3C/Adobe normative math | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |
| Vivid Light | GO for stated bounded operator | Qualified GO — canonical interior; explicit W3C Burn/Dodge endpoint extension | QUALIFIED GO — isolated API plus explicit caller-only Engine/grading route; no recipe wiring | UNVERIFIED |

“SPECIFICATION VERIFIED” here is a local source/code/test audit of the specified operator, not W3C certification and never Photoshop certification. For the last two modes, the qualification is essential: Adobe describes their effects but does not normatively specify the equations or singular-corner precedence used here.

## Acceptance audit and changes

The starting harness already had reduced analytical functions, an independent two-input implementation, domain hypotheses, fixtures and LUT experiments. Those functions were retained. The missing work was directory discovery without a case manifest, permissive numeric comparison with strict provenance downgrade, bounded mismatch locations, exhaustive specification tests, documented derivations, 16-bit domain experiments, and a capture script.

Changes close those offline gaps. Small numerical corrections are explicit: Burn/Dodge saturate before potentially unsafe division; nearest-half-up final quantization tolerates one downward arithmetic ULP at a half-DN tie. Neither correction fits historical screenshots. The `retouch/self_blend.py` module now hosts the analytical primitive, with caller-only adapters in `ColorGrader` and `RetouchEngine.process`; it is deliberately not enabled by recipes or UI defaults.

## 1. Model and complete derivations

Let `b = Cb`, `s = Cs`, each in `[0,1]`, in one explicitly selected RGB component space. `clip(t) = min(1,max(0,t))`. Layers are opaque and pixel-aligned. For a separable operator, substituting `b=s=x` produces a per-channel scalar function exactly over real arithmetic. This says nothing by itself about Photoshop's working space, intermediate quantization or compositor.

The eight W3C definitions below were checked against the normative blend sections of the **Compositing and Blending Level 1 Candidate Recommendation Draft, 21 March 2024**. This is a draft specification for that compositing model, not a normative specification of Photoshop. [W3C blend definitions](https://www.w3.org/TR/2024/CRD-compositing-1-20240321/#blending)

### Multiply

`B(b,s) = bs`.

Substitution gives `f(x)=x²`. Endpoints: `f(0)=0`, `f(1)=1`. The entire normalized interval is already in range; no clipping or singularity.

### Screen

`B(b,s)=1-(1-b)(1-s)=b+s-bs`.

Therefore `f(x)=1-(1-x)²=2x-x²`. Endpoints 0 and 1; no clipping or singularity.

### Overlay

`B(b,s) = 2bs` when `b≤1/2`; otherwise `1-2(1-b)(1-s)`.

Substitution gives:

\[
f(x)=\begin{cases}2x^2&0\le x\le1/2\\1-2(1-x)^2&1/2<x\le1.\end{cases}
\]

The branches meet at `f(1/2)=1/2`; endpoints are 0 and 1. No added clipping.

### Hard Light

`B(b,s) = 2bs` when `s≤1/2`; otherwise `1-2(1-b)(1-s)`.

Setting `s=b=x` makes the branch selection identical to Overlay, hence the same piecewise self-function and endpoints. **Overlay and Hard Light are interchangeable as self-transfer functions, not as general two-input operators.** At `(b,s)=(1/4,3/4)`, Overlay is `3/8`, Hard Light is `5/8`. Some off-diagonal inputs also coincide, but equality of the complete operators does not follow.

### Soft Light

\[
B(b,s)=\begin{cases}
b-(1-2s)b(1-b)&s\le1/2\\
b+(2s-1)(D(b)-b)&s>1/2,
\end{cases}
\quad
D(b)=\begin{cases}((16b-12)b+4)b&b\le1/4\\\sqrt b&b>1/4.\end{cases}
\]

For the self lower branch, expand `x-(1-2x)x(1-x)=3x²-2x³`. For the self upper branch, `s=b=x>1/2` necessarily implies `b>1/4`, so only the square-root branch of `D` is reachable:

\[
f(x)=\begin{cases}x^2(3-2x)&x\le1/2\\x+(2x-1)(\sqrt x-x)&x>1/2.\end{cases}
\]

Endpoints 0 and 1; midpoint 1/2. It is continuous but the self-function's derivative changes at 1/2. The polynomial `D` branch is **not dispensable in the independent two-input reference**: tests include small backdrops with `s>1/2`, plus neighbors of `b=1/4`. No denominator singularity or added clipping.

### Color Dodge

Endpoint precedence is part of the definition:

\[
B(b,s)=\begin{cases}0&b=0\\1&b\ne0,\ s=1\\\min(1,b/(1-s))&\text{otherwise}.\end{cases}
\]

In particular `B(0,1)=0`, not 1. For `s<1`, saturation begins at `b≥1-s`. Along the self diagonal, `x/(1-x)≥1` iff `x≥1/2`:

\[
f(x)=\begin{cases}x/(1-x)&0\le x<1/2\\1&1/2\le x\le1.\end{cases}
\]

Thus `f(0)=0`, `f(1)=1`. The singular expression at `x=1` is never evaluated. The reference implementation also detects saturation before division, avoiding overflow with tiny denominators.

### Color Burn

\[
B(b,s)=\begin{cases}1&b=1\\0&b\ne1,\ s=0\\1-\min(1,(1-b)/s)&\text{otherwise}.\end{cases}
\]

Hence `B(1,0)=1`, not 0. For `b≠1,s>0`, zero saturation holds when `s≤1-b`. On the diagonal, `(1-x)/x≥1` iff `x≤1/2`; otherwise `1-(1-x)/x=2-1/x`:

\[
f(x)=\begin{cases}0&0\le x\le1/2\\2-1/x&1/2<x\le1.\end{cases}
\]

`f(0)=0`, `f(1)=1`; division at zero is never evaluated. The two-input implementation divides only where the unsaturated ratio is below one.

### Exclusion

`B(b,s)=b+s-2bs`, giving `f(x)=2x(1-x)`.

Endpoints are both zero; the maximum is `f(1/2)=1/2`. The function increases then decreases. It is not invertible over `[0,1]`, and must not be forced through a monotonic spline. No clipping or singularity is needed.

### Linear Light — bounded canonical candidate

Use `B(b,s)=clip(b+2s-1)`. Below or at `s=1/2`, this is bounded linear burn with doubled source; above it, it is bounded addition of `2s-1`. Both expressions reduce to the same line before clipping. The general clipping regions are `b+2s≤1` (zero) and `b+2s≥2` (one).

\[
f(x)=\operatorname{clip}(3x-1)=\begin{cases}0&x\le1/3\\3x-1&1/3<x<2/3\\1&x\ge2/3.\end{cases}
\]

Midpoint 1/2, endpoints 0 and 1, no singularity. This is the **bounded normalized** canonical operator; an unbounded/HDR formulation is not being silently substituted. The canonical arithmetic is supported by GIMP's author documentation; Adobe's brightness-based description is qualitative. [GIMP contrast-mode definitions](https://docs.gimp.org/3.0/en/layer-mode-group-contrast.html), [Adobe mode descriptions](https://helpx.adobe.com/ca/photoshop/desktop/repair-retouch/adjust-light-tone/blending-mode-descriptions.html)

### Vivid Light — bounded canonical candidate

Define the general operator through the endpoint-safe Burn/Dodge above:

\[
B(b,s)=\begin{cases}\operatorname{Burn}(b,2s)&s\le1/2\\\operatorname{Dodge}(b,2s-1)&s>1/2.\end{cases}
\]

For `0<s≤1/2` away from `b=1`, the lower expression is `max(0,1-(1-b)/(2s))`, clipping to zero when `b+2s≤1`. At `s=0`, it is zero except `B(1,0)=1` under the chosen Burn precedence. For `1/2<s<1`, the upper expression is `min(1,b/[2(1-s)])`, clipping to one when `b+2s≥2`. At `s=1`, it is one except `B(0,1)=0` under the chosen Dodge precedence. At `s=1/2` the result is `b`.

Substitute `b=s=x`. The lower branch exceeds zero when `3x>1` and simplifies to `(3x-1)/(2x)`; the upper branch reaches one when `3x≥2`:

\[
f(x)=\begin{cases}
0&0\le x\le1/3\\
(3x-1)/(2x)&1/3<x\le1/2\\
x/[2(1-x)]&1/2<x<2/3\\
1&2/3\le x\le1.
\end{cases}
\]

The joins are 0, 1/2 and 1. Singular endpoint expressions are never evaluated. Vivid Light and Linear Light **share their self clipping intervals** but differ inside `(1/3,1/2)` and `(1/2,2/3)`: at `x=.4`, Vivid=.25 and Linear=.20; at `.6`, Vivid=.75 and Linear=.80. They also meet at `.5`.

GIMP's author documentation supplies the interior formulas and bounds. It does not settle the `0/0` corners; P5 explicitly extends them with W3C Burn/Dodge precedence. This is a tested, well-defined canonical candidate, **not proof of Adobe's endpoints or implementation**. [GIMP Vivid Light description](https://docs.gimp.org/3.0/en/layer-mode-group-contrast.html)

## 2. Source/implementation evidence ledger

`blend(b,s,mode)` is independent of `self_transfer(x,mode)`. Tests add a third, scalar transcription of each stated definition and evaluate **all 65,536 two-input 8-bit pairs per mode**, not just the diagonal. Dense self tests use both float64 and float32; singular/nextafter tests additionally exercise the critical neighborhoods.

| Mode | Analytical derivation | Reference definition checked | Independent two-input implementation | Self-reduction equality tested | Status |
|---|---|---|---|---|---|
| Multiply | substitution → x² | W3C §10.1.2 | `blend`; scalar product oracle | PASS | W3C definition verified |
| Screen | complement expansion | W3C §10.1.3 | `blend`; complement-form scalar oracle | PASS | W3C definition verified |
| Overlay | branch on backdrop | W3C §10.1.4 | swapped Hard Light; explicit backdrop-branch scalar oracle | PASS + off-diagonal distinction | W3C definition verified |
| Hard Light | branch on source | W3C §10.1.9 | two branches; scalar source-branch oracle | PASS | W3C definition verified |
| Soft Light | cubic / square-root reduction | W3C §10.1.10 including D polynomial | complete two-input D; expanded scalar polynomial oracle | PASS | W3C definition verified |
| Color Dodge | ratio and saturation | W3C §10.1.7 including b=0 precedence | saturation-before-division; literal scalar branches | PASS | W3C definition verified |
| Color Burn | ratio and saturation | W3C §10.1.8 including b=1 precedence | safe unsaturated division; literal scalar branches | PASS | W3C definition verified |
| Exclusion | quadratic reduction | W3C §10.1.12 | sum/product two-input expression; scalar oracle | PASS + descending intervals | W3C definition verified |
| Linear Light | clipped affine reduction | GIMP author arithmetic; P5 normalized bounds; Adobe descriptive only | clipped b+2s−1; scalar oracle | PASS | Canonical bounded definition verified; not normative Adobe |
| Vivid Light | Burn/Dodge piecewise reduction | GIMP interior/bounds; P5 W3C endpoint extension; Adobe descriptive only | actual Burn/Dodge branches; scalar oracle | PASS + distinction from Linear | Qualified canonical verification; corner extension explicitly chosen |

Sources were stopped once these definitions and the remaining evidence gaps were resolved. No generic beauty, generative retouching or P4 research is included.

## 3. Opacity and domain analysis

### What opacity is proved

For source-over blending, with opaque backdrop `αb=1`, layer opacity `αs=p`, and no masks/effects, the output alpha is `p+(1-p)=1`. The source color entering compositing is `B(b,s)`, so the resulting unpremultiplied component is:

\[
R=(1-p)b+pB(b,s),\qquad R_{self}=(1-p)x+pf(x).
\]

This is algebra in the **selected compositing domain**. Tests cover 25/50/75/100%, opacity-zero identity and opacity-one operator identity for all ten modes and all four domain hypotheses. A simple spatial mask is only multiplication `p_eff=p·mask`; mask-zero is exact identity. It is not a test of Photoshop's mask subsystem. [W3C source-over](https://www.w3.org/TR/2024/CRD-compositing-1-20240321/#porterduffcompositingoperators_srcover)

Layer **Opacity is not Fill**. The capture keeps Fill=100. This experiment does **not** establish Fill-specific behavior, Blend If, masks beyond simple alpha/opacity multiplication, transparency-compositing edge cases, layer effects, group blending, or HDR/32-bit semantics. Adobe documents separate Opacity/Fill controls; that does not prove their numerical equivalence. [Adobe opacity and blending](https://helpx.adobe.com/uk/photoshop/using/layer-opacity-blending.html)

For Exclusion at partial opacity, `R=(1+p)x−2px²`; derivative `1+p−4px` becomes negative on part of the interval when `p>1/3`. Thus 25% is monotonic but 50/75/100% are not. The white endpoint is `1-p`.

### Four explicit domain hypotheses

Let `L` decode sRGB and `E=L⁻¹` encode it. For normalized values:

`L(x)=x/12.92` for `x≤.04045`, otherwise `((x+.055)/1.055)^2.4`.

`E(t)=12.92t` for `t≤.0031308`, otherwise `1.055t^(1/2.4)−.055`.

These are piecewise sRGB transfers, not a blanket gamma 2.2 approximation. Their tiny published breakpoint round-trip discrepancy is covered by tolerance. [W3C sRGB space and transfer](https://www.w3.org/TR/css-color-4/#predefined-sRGB)

| Harness model | Encoded output before final quantization |
|---|---|
| `encoded` | `(1-p)x+p f(x)` |
| `linear` | `E((1-p)L(x)+p f(L(x)))` |
| `encoded_blend_linear_opacity` | `E((1-p)L(x)+p L(f(x)))` |
| `linear_blend_encoded_opacity` | `(1-p)x+p E(f(L(x)))` |

In code, transfer encode/decode round trips can introduce floating-point error at machine precision; tests bound it. The two hybrid models intentionally separate the blend kernel's domain from opacity interpolation's domain. At p=1 each hybrid collapses to its corresponding blend-domain hypothesis, so that opacity alone cannot identify the compositor domain.

Adobe documents blending directly in document space when **Blend RGB Colors Using Gamma** is disabled, and a gamma-associated compositing space when enabled. Its example concerns Normal blending; this is insufficient to infer how every non-Normal blend kernel interacts with that control. [Adobe advanced color settings](https://helpx.adobe.com/photoshop/using/color-settings.html)

Working-space profile, embedded image profile, import policy and blend-gamma preferences are separate facts. “Encoded RGB” is not universally “sRGB.” P5 performs no hidden ICC conversion. `--transfer gamma --gamma VALUE` is an explicit power-law hypothesis, not an ICC implementation or a Photoshop certificate. Arbitrary working-space conversions may mix channels; a scalar LUT in one working space is not automatically the same LUT in another. No `photoshop_compatible` API/domain alias is justified yet.

### Measured domain disagreements — not Photoshop observations

The experiment evaluates every input code on grayscale ramps (256 and 65,536 pixels) and on the RGB atlas at both depths. It records 480 domain-comparison rows: 10 modes × 4 opacities × 3 alternatives to encoded × 2 depths × 2 sample sets. Each includes max error, MAE, RMSE, pixel/channel counts >1 DN, pixel fraction, worst-input coordinates and bounded ties.

The table shows **linear versus encoded worst absolute DN disagreement**, in opacity order **25%, 50%, 75%, 100%**, on exhaustive grayscale ramps. This is an analytical-domain difference after the declared nearest-half-up quantization, not error against Photoshop:

| Mode | 8-bit max DN at four opacities | 16-bit max DN at four opacities |
|---|---|---|
| Multiply | 5, 7, 5, 8 | 1057, 1607, 1266, 1910 |
| Screen | 6, 12, 19, 27 | 1396, 3040, 4877, 6869 |
| Overlay | 12, 24, 36, 50 | 2987, 6062, 9255, 12611 |
| Hard Light | 12, 24, 36, 50 | 2987, 6062, 9255, 12611 |
| Soft Light | 7, 15, 23, 32 | 1798, 3698, 5741, 8010 |
| Color Dodge | 28, 56, 84, 112 | 7180, 14397, 21647, 28927 |
| Color Burn | 17, 38, 69, 162 | 4291, 9872, 18044, 41949 |
| Exclusion | 34, 62, 87, 112 | 8574, 15762, 22320, 28584 |
| Linear Light | 34, 72, 118, 213 | 8628, 18346, 30253, 54885 |
| Vivid Light | 30, 65, 109, 201 | 7855, 16800, 27934, 51793 |

The worst listed 8-bit case is Linear Light 100%: 213 DN at input 156; MAE 47.684, RMSE 79.233, 127/256 pixels >1 DN (49.609%). At 16-bit it is 54,885 DN at input 40,140; MAE 12,306.285, RMSE 20,405.119, 32,942/65,536 pixels >1 DN (50.266%). These large differences justify keeping the processing domain explicit. They do not tell us which domain Photoshop uses.

Complete per-mode/per-opacity MAE, RMSE, counts/fractions and mismatch/worst-input locations, including RGB contexts and hybrid hypotheses: [domain CSV](../../test_output/p5_self_blend_20260906/analysis_final/domain_differences.csv) and [full analytical JSON](../../test_output/p5_self_blend_20260906/analysis_final/analytical_validation.json).

## 4. Historical reference points

The supplied diagrams are historical sanity checks, not specifications. This task had their transcribed points, not a fresh set of actual Photoshop exports; nothing was fitted to screenshots.

| Mode / input DN | Analytical output DN before encoding quantization | Nearest half up | Floor | Interpretation |
|---|---:|---:|---:|---|
| Multiply 64 | 16.062745 | 16 | 16 | Supports the approximate point |
| Screen 192 | 239.435294 | 239 | 239 | **Not 247**; discrepancy remains, not “corrected” by fitting |
| Overlay / Hard Light 64 | 32.125490 | 32 | 32 | Supports the approximate point |
| Overlay / Hard Light 128 | 128.498039 | 128 | 128 | Rounds to 128; normalized value is not exactly 1/2 |
| Overlay / Hard Light 192 | 223.870588 | 224 | 223 | Rounding policy matters |
| Soft Light 51 | 26.520000 | 27 | 26 | Diagram 26 is compatible with truncation, not nearest rounding |
| Soft Light 128 | 128.206531 | 128 | 128 | Approximate midpoint only |
| Color Dodge 85 | 127.500000 | 128 | 127 | Exact real-arithmetic half-DN tie |
| Color Dodge 128 | 255 | 255 | 255 | Already in clipped upper half |
| Color Burn 128 | 1.992188 | 2 | 1 | **Not exactly zero** |
| Color Burn 170 | 127.500000 | 128 | 127 | Half-DN tie |
| Exclusion 128 | 127.498039 | 127 | 127 | Diagram 128 describes the ideal midpoint approximately |
| Linear Light 128 | 129 | 129 | 129 | **Not 128** with x=128/255 |
| Vivid Light 128 | 128.503937 | 129 | 128 | Distinct from Linear Light before quantization |

`128/255>.5`; likewise `32768/65535>.5`. The exact midpoint would be code 127.5 or 32767.5, not an integer input. Burn and Dodge have different branch membership at codes 127 and 128. Linear/Vivid clip at 85/170 in 8-bit and 21845/43690 in 16-bit, with different interior values.

The saved reference CSV explicitly includes 84/85/86, 127/128, 169/170/171 and useful quarter/third/half/two-third 16-bit neighbors. Float64/float32 nextafter tests probe below/at/above 0, 1/4, 1/3, 1/2, 2/3 and 1 where inside the legal interval. Vivid Light receives dedicated three-way probes around 1/3, 1/2, 2/3.

## 5. Golden workflow — no case manifest required

Generate baselines once:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qa/p5_self_blend_experiment.py prepare test_output/p5_fixtures
```

The generated sRGB-tagged `rgb_atlas_8.png` and `rgb_atlas_16.png` have deterministic samples, exhaustive gray and pure R/G/B ramps, channel-context tiles, mixed RGB and near-black/near-white ramps. Their native dimensions are 1024×160 and 1024×640. PNG round-trip tests preserve all uint8/uint16 RGB samples. ICC creation timestamps can vary between runs; recorded file/profile hashes identify the actual fixture rather than promising timestamp-independent file bytes.

For each Photoshop configuration and depth, use a **separate directory**:

```text
test_data/photoshop_self_blend/
    baseline.png
    normal_100.png                      # needed for a parity claim, optional to score
    multiply_025.png
    multiply_050.png
    multiply_075.png
    multiply_100.png
    ...
    vivid_light_100.png
    capture_metadata.json              # OR external_golden_metadata.json; optional to score
    settings_evidence.png              # only required for the provenance gate
```

TIFF (`.tif`/`.tiff`) works too. Mode stems are the lowercase harness names; opacity suffixes are exactly `025`, `050`, `075`, `100`. Do not resize, convert the output profile, rotate, use JPEG, or mix configurations. The baseline must be the actual input canvas of those exports, not a separately reconstructed ramp. Duplicate PNG/TIFF cases for the same mode/opacity are rejected.

**One-command import and comparison:**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qa/p5_self_blend_experiment.py compare test_data/photoshop_self_blend test_output/p5_photoshop_comparison
```

No metadata and no 40-case JSON are needed for numeric analysis. A partial set is scored and reports every missing pair. Malformed metadata JSON is retained as an error and cannot attest provenance, but filename-discovered images remain scoreable. Add `--require-complete` to reject incomplete image sets. Use a new output path on reruns; the harness deliberately refuses overwriting prior evidence.

Outputs are `photoshop_comparison.csv` and `.json`, with one row per mode/opacity/**domain hypothesis**/rounding policy. A complete set produces 320 rows. Each contains:

- Max absolute DN error, MAE, RMSE; pixel count and fraction >1 DN; channel samples >1 DN.
- First ten nonzero mismatches, first ten >1 DN mismatches, and first ten locations attaining maximum error, plus the total number of maximum-error ties.
- For each location: x, y, RGB channel, baseline/input DN, predicted DN, actual DN, and signed error `predicted−actual`.

Locations use row-major y/x/RGB order. A pixel counts once if any channel exceeds 1 DN. DN values are integer code units of the exported bit depth, not all scaled to 8-bit. Equal arrays have empty mismatch/max-location lists. JSON does not contain an unbounded list of individual mismatches.

Optional `--save-error-images` also produces signed int32 `.npy` errors and PNG displays centered at 128 with 16× DN amplification. Display clipping is visualization only; numeric metrics use the original signed error. No display-space screenshot is treated as a pixel oracle.

### Metadata and evidence gates

The capture script generates metadata automatically. For independently exported Photoshop images, copy [the small metadata template](../../scripts/qa/p5_external_golden_metadata.example.json) as `external_golden_metadata.json` and fill actual facts. There is no `cases` list to maintain. `baseline` and `normal_control` may optionally override their default filenames.

The gate requires an attested Photoshop producer, complete capture status, actual version, document profile, bit depth, operator, capture notes, known blend-gamma enablement/value, confirmed settings with a local evidence attachment, Fill=100, opaque equal layers, disabled Blend If, no effects/masks, exhaustive gray and R/G/B input coverage, all 40 pairs, consistent embedded ICC profiles, and an unchanged Normal control. When the capture script includes the original fixture, its samples/profile must survive import/baseline export unchanged. This catches otherwise hidden import conversions or code-value changes.

Unknown settings, absent metadata/profile/control, missing cases or ICC/control mismatches **downgrade**, rather than suppress, compatible-image comparisons. Invalid dimensions/bit depth/channels are rejected: there is no silent resampling. Missing baseline, duplicate/malformed cases or unsafe asset paths are input errors. Two competing metadata files require explicitly selecting the intended JSON.

Only a fully attested set where **one consistent domain model and rounding policy passes all 40 cases with max absolute error ≤1 DN** can obtain `PHOTOSHOP_VERIFIED_CAPTURE_SCOPE_ONLY`. No picking a different model per mode or per pixel. At present all real-world parity is UNVERIFIED. Without valid provenance the label remains `EXTERNAL_IMAGE_COMPARISON_PHOTOSHOP_UNVERIFIED`, even with zero numeric error.

Attestation and attached evidence need human review: JSON is not cryptographic proof that Adobe produced the pixels. Full-profile conversion is not implemented; non-sRGB/power-gamma hypotheses cannot be promoted by this version's gate. Strict ICC-byte matching may conservatively downgrade an equivalent reserialized profile; review that mismatch, do not silently ignore it. A passing capture is limited to its recorded version/configuration/depth and does not by itself explain Photoshop's internals or authorize production integration.

### Photoshop capture script audit

[p5_photoshop_capture.jsx](../../scripts/qa/p5_photoshop_capture.jsx) uses Photoshop's File > Scripts > Browse workflow. It opens a generated fixture, makes fresh duplicate documents and duplicate layers, applies each of the ten requested modes at 25/50/75/100% **layer Opacity**, reads the requested layer settings back, leaves Fill=100 and exports flattened, uncompressed TIFF with the document ICC embedded. It also exports a baseline, a Normal-100 duplicate control, the original fixture, and capture metadata.

It records Photoshop version, document profile, bit depth/dimensions and the accessible Color Settings preset name. The preset name is **not** treated as proof of blend gamma. Reliable blend-gamma access is unavailable in this DOM implementation: defaults are UNKNOWN/UNAVAILABLE. An operator can attest the inspected setting and attach evidence; otherwise comparisons remain unverified. It does not guess Action Manager keys or change global color preferences. When opening prompts about profiles, preserve the embedded profile; the later source-fixture check detects changes.

Exports go to a new timestamped directory. The script refuses to reuse an already-open fixture, never edits an unrelated document, restores dialog/active-document state, closes only its own documents without saving originals, and records failed/incomplete captures without discarding partial output. `TiffSaveOptions`, `Document.saveAs`, layer duplication/opacity/fill, profile/bit-depth properties and enum names were audited against Adobe's JavaScript reference. [Adobe Photoshop JavaScript reference, including TiffSaveOptions p.188](https://github.com/Adobe-CEP/CEP-Resources/blob/master/Documentation/Product%20specific%20Documentation/Photoshop%20Scripting/photoshop-cc-javascript-ref-2019.pdf)

**Verification boundary:** Node mock execution checked parsing/control flow, 42 lossless-export requests, all 40 unique mode/opacity cases, Fill=100, metadata, UNKNOWN gamma and document cleanup. It has no Photoshop renderer. No claim is made that the installed Photoshop DOM, TIFF encoder, gamma behavior or 16-bit internals have been runtime-verified. The first real capture must check those observations.

## 6. Test results and retained artifacts

The final focused suite contains **113 passing tests**. It covers every requested analytical identity/non-identity, all 8-bit two-input pairs, float32/64 dense reductions, singular precedence, Soft Light's lower-backdrop polynomial, boundaries, 8-/16-bit midpoints, opacity endpoints and four domains, simple mask zero/fractional behavior, input immutability, out-of-range/nonfinite rejection, off-grid LUT errors, deterministic fixture round trips, filename discovery, incomplete/malformed cases, provenance downgrade, baseline coverage, signed mismatch coordinates, optional 16-bit error outputs, source/control failures, and the JSX mock.

Numerical findings from the standalone validation:

- 65,551 dense/boundary inputs per mode; largest float64 self-vs-two-input discrepancy **2.212×10⁻¹⁶**, consistent with arithmetic rounding, not a mathematical disagreement.
- Largest float32 self error against float64: **0.0000422 DN at 8-bit scale**, approximately **0.01085 DN at 16-bit scale** on these samples. This is an offline precision check, not production qualification.
- All required opacity values agree with the independent two-input compositing calculation within the asserted tolerance. No NaN, Inf or divide-by-zero; valid subnormal products may underflow to zero.
- 4096-entry uniformly sampled LUTs are not automatically 16-bit-safe: Burn/Dodge reach approximately **15.9998 DN** interpolation error at 16-bit scale near the half-range kink. 65,536 entries reduce that sampled error to **0.999985 DN**; this alone does not guarantee post-quantization parity. Soft Light's corresponding maxima are **0.34351** and **0.02145 DN**. Linear Light's third/two-third knots happen to align with both chosen grids.
- Direct vectorized equations are the simplest correctness reference. If LUTs are later used, include analytical breakpoints and test off-grid interpolation after the complete domain/opacity mapping; a 4096-node preview LUT is not a general high-quality production guarantee.

Actual commands executed (repository root):

```bash
PYTHONPYCACHEPREFIX=/private/tmp/p5_pycache .venv/bin/python -m py_compile scripts/qa/p5_self_blend_experiment.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_p5_self_blend_experiment.py
node tests/fixtures/p5_capture_mock.cjs scripts/qa/p5_photoshop_capture.jsx
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qa/p5_self_blend_experiment.py prepare test_output/p5_self_blend_20260906/fixtures
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qa/p5_self_blend_experiment.py validate test_output/p5_self_blend_20260906/analysis_final
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qa/p5_self_blend_experiment.py compare test_output/p5_self_blend_20260906/synthetic_import_smoke test_output/p5_self_blend_20260906/synthetic_comparison --require-complete
```

The import smoke images were produced by the test fixture helper (`synthetic_set(..., full=True)`), not Photoshop. All 40 encoded/nearest predictions match those synthetic files exactly, while the evidence label correctly stays EXTERNAL IMAGE COMPARISON / PHOTOSHOP UNVERIFIED. This validates the CLI/discovery/metrics path only; it is not independent photographic evidence. The test suite also deliberately supplies mock attestation to exercise the positive gate, exclusively inside temporary test directories; those files are not real Photoshop goldens.

An initial compile attempt encountered macOS's external bytecode-cache write restriction; rerunning with the explicit temporary cache succeeded. Early test development caught an incorrect Overlay midpoint expectation, a scalar half-DN rounding edge case and a Node mock variable-name conflict; these were resolved before the final pass, without modifying blend equations to match a diagram.

Native-resolution artifacts (all analytical, not Photoshop exports):

- [8-bit baseline atlas](../../test_output/p5_self_blend_20260906/fixtures/rgb_atlas_8.png), [16-bit baseline atlas](../../test_output/p5_self_blend_20260906/fixtures/rgb_atlas_16.png), [fixture manifest](../../test_output/p5_self_blend_20260906/fixtures/fixtures.json).
- [Analytical curve/ramp viewer](../../test_output/p5_self_blend_20260906/analysis_final/analytic_review.html): native 1024×256 predicted PNGs plus exact-equation sampled SVG curves, all four opacities.
- [Full validation JSON](../../test_output/p5_self_blend_20260906/analysis_final/analytical_validation.json), [domain metrics CSV](../../test_output/p5_self_blend_20260906/analysis_final/domain_differences.csv), [8-/16-bit boundary/reference points](../../test_output/p5_self_blend_20260906/analysis_final/reference_points.csv), [LUT errors](../../test_output/p5_self_blend_20260906/analysis_final/lut_errors.csv).
- [Synthetic-only importer smoke JSON](../../test_output/p5_self_blend_20260906/synthetic_comparison/photoshop_comparison.json), [CSV](../../test_output/p5_self_blend_20260906/synthetic_comparison/photoshop_comparison.csv).

## 7. Architecture recommendation and remaining evidence

Retain the analytic function as the reference primitive; a future self-transfer API can explicitly select mode, blend space, compositing space, opacity, rounding and precision. Overlay/Hard Light may alias at the self-transfer level only. Keep Vivid/Linear distinct. Avoid hand-authored Bézier control points and monotonic enforcement. Keep profile conversion outside the kernel with an explicit contract. Nonlinear per-channel operations can alter hue/saturation, collapse tones and amplify existing quantization; they are not luminance-only or local-contrast operations. A pointwise curve itself does not spatially convolve pixels or create a convolution halo.

Before any production proposal:

1. Run the JSX in actual Photoshop with the supplied 8-bit sRGB atlas; inspect metadata, Normal control, profile and layer settings. Then repeat 16-bit separately. Record and investigate any import-code-value changes instead of declaring them rounding noise.
2. Capture distinct known blend-gamma configurations (off and checked 1.00 first), using a fixed profile and same fixture. Require all 40 cases at each configuration. Add further working spaces only after implementing their transfer/profile handling explicitly.
3. Review all mismatch summaries and worst-input locations. A candidate model/rounding must pass the declared ≤1 DN tolerance consistently across the full capture, with trustworthy provenance. Do not tune per-mode ad hoc curves to get a pass; explain any changed arithmetic from independent evidence.
4. Repeat on a second capture/version or otherwise scope the compatibility claim narrowly. Add native 16-bit gradients and representative photographic color checks before product exposure. Quantitative ramp parity alone is not a portrait-quality recommendation.
5. Only then request separate authorization for recipe/UI exposure and
   performance/memory measurements. Existing RetouchEngine recipe behavior has
   intentionally not been changed by P5.

Research/tooling and isolated API files in this change: this report;
`retouch/self_blend.py`; `scripts/qa/p5_self_blend_experiment.py`;
`scripts/qa/p5_photoshop_capture.jsx`;
`scripts/qa/p5_external_golden_metadata.example.json`;
`tests/test_self_blend.py`; `tests/test_p5_self_blend_experiment.py`;
`tests/fixtures/p5_capture_mock.cjs`. Generated evidence remains under the
P5-specific `test_output` directory. Existing recipe behavior and P4's
accepted research remain untouched.

Original research verification: focused pytest **113 passed in 1.09s**; Python compilation passed; JSX mock passed; prepare, final validation and complete synthetic directory comparison exited successfully. All research source/report/test files passed whitespace/diff checks, report-local artifact links resolved, final validation's runner hash matched the harness, and P4's SHA-256 remained `f8b1109537e9ee343fb4a310c1b1fba30e87f50bb331e8f89f27b136afc7940f`. The later isolated API follow-through is covered below; no files were staged or committed.

## 8. Production follow-through (2026-09-07)

`retouch/self_blend.py` now provides the stable, explicitly analytical
`self_blend_transfer(value, mode)` and `apply_self_blend(image, mode,
amount=..., domain=...)` APIs. All ten derived operators are vectorized and
endpoint-safe, with `amount` representing opaque-layer opacity in the selected
encoded or linear sRGB hypothesis. Float inputs preserve their floating dtype;
uint8/uint16 output uses an explicit nearest-half-up policy. Exclusion remains
non-monotonic by construction, and no spline/LUT assumption is introduced.

`ColorGrader.apply_self_blend_tone(...)` is the production grading adapter, and
`RetouchEngine.process(...)` accepts caller-only `self_blend_mode`,
`self_blend_amount`, and `self_blend_domain` arguments. The operation is applied
only when a caller selects a mode; no recipe or UI default supplies it.

`tests/test_self_blend.py` covers finite/bounded output, independent
two-input/self-diagonal equality, endpoints and branch boundaries, Screen DN
192, Overlay/Hard Light self-equivalence and off-diagonal distinction,
Vivid/Linear distinction, encoded-vs-linear behavior, amount-zero and
25/50/75/100% opacity semantics, and singular endpoint safety. The focused command
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_self_blend.py
-p no:cacheprovider` passes **14 tests**. This validates the analytical module
and explicit caller-only route, not existing recipe behavior or Photoshop
rendering. There is no production recipe/UI enablement and no
`photoshop_compatible` alias.

**Final decision: GO the isolated analytical operator infrastructure and
offline evidence; NO-GO Photoshop pixel parity, Photoshop Fill claims, a
Photoshop-compatible domain label, or automatic recipe/pipeline migration
until provenance-valid goldens and a separate integration review exist.**
