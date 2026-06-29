# VISUAL_QA.md — Visual Quality Assurance Gates
> Load when: any change touches a pipeline stage. Visual QA is mandatory for all pipeline-stage modifications.
> Bypass is only permitted for documentation-only changes and non-pipeline modules (params.py, io.py, recipe_schema.py).

---

## 1. The Eight Gates

| Gate | Criterion | How to Test | Automatic FAIL |
|------|-----------|-------------|----------------|
| **Texture Preservation** | Pore structure visible at 100% zoom in skin region | Extract high-freq band from `frequency.py` before and after change; compare SSIM on skin patch | High-freq band SSIM < 0.92 on skin patch |
| **No Halo Artifacts** | No luminance ring around face edges, feature edges, or grading boundaries | Inspect at 100% zoom: facial silhouette, lip edge, eye contour, hair boundary | Any visible luminance ring at any edge |
| **No Edge Tearing** | Geometry warp produces smooth, continuous transitions | Diff warp displacement field for discontinuities; inspect hairline and jaw boundary at 100% | Any pixel displacement discontinuity > 2px |
| **No Color Drift** | Neutral gray regions stay neutral after all processing | Sample 3 neutral gray patches (background, garment, catchlight); measure ΔE (LAB) before/after | ΔE > 2.0 on any neutral gray sample |
| **No Highlight Clipping** | White garment and specular detail preserved | Check luminance histogram top 5% percentile; no flat plateau at 255 | > 0.5% of pixels saturate to 255 in final output |
| **No Shadow Crushing** | Dark hair and background shadow retain detail | Check luminance histogram bottom 5% percentile; no flat floor at 0 | > 0.5% of pixels clip to 0 in final output |
| **Skin Tone Uniformity** | Tone correction is spatially uniform; no patchy equalization | Visual inspection of skin region; compare to reference portrait at uniform illumination | Visible patchwork or mask-boundary artifact in skin |
| **Natural Output** | Output looks clean at 100% zoom; no uncanny valley | Full image review at 100%; subjective judgment by any reviewer | Any reviewer identifies "plastic skin", artificial look, or processing artifact |

---

## 2. Gate Scope by Module

Apply only the gates relevant to the modified stage. Use the blast radius table in `docs/PIPELINE.md` to determine which downstream stages are also in scope.

| Module changed | Required gates |
|---------------|---------------|
| `detection.py` | All eight gates |
| `parsing.py` | Skin Tone Uniformity, No Halo |
| `geometry.py` | No Edge Tearing, Natural Output |
| `frequency.py` | Texture Preservation, No Halo, Natural Output |
| `skin.py` | Texture Preservation, Skin Tone Uniformity, Natural Output |
| `eyes.py` | Natural Output (isolated eye region) |
| `lips.py` | Natural Output (isolated lip region) |
| `teeth.py` | Natural Output (isolated teeth region) |
| `blemish.py` | Texture Preservation, Natural Output |
| `style.py` | No Color Drift, Natural Output |
| `grading.py` | No Color Drift, No Highlight Clipping, No Shadow Crushing, No Halo |
| `params.py` | All eight gates (full regression) |

---

## 3. Reporting Visual QA Results

All gate results must appear in `VERIFY_OUTPUT.visual_qa`. Format:

```
visual_qa: [
  "Texture Preservation: PASS — SSIM 0.96 on skin patch",
  "No Halo: PASS — no luminance ring at facial silhouette",
  "No Color Drift: PASS — ΔE 1.2 on gray patch",
  "No Highlight Clipping: PASS — 0.1% pixel saturation",
  "No Shadow Crushing: PASS — 0.0% pixel floor",
  "Skin Tone Uniformity: PASS — uniform correction, no boundary artifact",
  "Natural Output: PASS — clean at 100% zoom"
]
```

"Looks good" is not a valid result. Describe what you inspected and what you found.
A gate not listed is treated as not tested — same as FAIL for protocol purposes.

---

## 4. Visual QA FAIL Protocol

A failing visual QA gate triggers the same Escalation Protocol as a failing unit test.

1. Document the gate, the failure description, and the attempted fix in `ESCALATION_REPORT.visual_qa_failures`.
2. After 2 consecutive fix attempts on the same gate without improvement: emit STUCK, stop retries.
3. Do not ship code that passes tests but fails a visual QA gate.

**Visual fidelity > correctness** (see `AGENTS.md` §0). A gate failure overrides a passing test suite.

---

## 5. Reference Images

Visual QA should be run against consistent reference images to make results comparable across changes.
Recommended reference set:
- A cosplay/costume portrait (primary — matches the project's target aesthetic)
- A studio portrait with neutral gray background (for No Color Drift gate)
- A high-key portrait with white garments (for No Highlight Clipping gate)
- A low-key portrait with dark background (for No Shadow Crushing gate)

If reference images are not available, document which were used and note the limitation in `VERIFY_OUTPUT.deferred`.