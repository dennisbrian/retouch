# PRECISION.md — Numerical Correctness Reference
> Load when: any change touches pixel arrays, colorspace conversions, masks, or kernel operations.
> All rules here are automatic CRITICAL findings in the Review Loop if violated.

---

## 1. Float32 Contract

```
PRECISION CONTRACT {
  internal_dtype:    float32
  input_conversion:  img.astype(np.float32) / 255.0     ← on first read
  output_conversion: np.clip(result * 255, 0, 255).astype(np.uint8)  ← at final write only
  forbidden:         uint8 arithmetic between pipeline stages
  forbidden:         implicit int promotion via np.uint8 array operations
}
```

**Why this matters**: `np.uint8(200) + np.uint8(100) = 44`. Silent wrap. No error.
The bug compiles, passes tests, and looks fine at thumbnail size. It shows at 100% zoom as color banding and tonal discontinuities. This is the most common invisible failure class in this codebase.

**Enforcement**: Any `np.uint8` or `np.int8` intermediate in pixel arithmetic is an automatic CRITICAL finding in the Review Loop. No exceptions.

---

## 2. Colorspace Contract

Every function operating on color values MUST declare its colorspace in its docstring.

```python
# Correct
def equalize_skin_tone(lab_img: np.ndarray) -> np.ndarray:
    """Input: float32 LAB. L∈[0,100], a∈[-128,127], b∈[-128,127]. Output: same."""

# Forbidden — colorspace is ambiguous
def process(img: np.ndarray) -> np.ndarray: ...
```

### Mandatory colorspace assignments

| Operation | Required colorspace | Reason |
|-----------|-------------------|--------|
| Skin tone equalization | **LAB** | BGR is perceptually non-uniform; produces uneven corrections |
| Luminance-only ops (dodge/burn) | **LAB, L-channel only** | Isolates luminance without touching chroma |
| Hue/saturation ops | **HSV or HLS** | Direct hue control |
| Frequency separation | **LAB** | Preserves perceptual uniformity across bands |
| LUT / grading ops | BGR (if LUT built for BGR) | Document explicitly |

### Boundary map — explicit `cv2.cvtColor` required at every crossing

```
OpenCV I/O   →  BGR
MediaPipe    →  RGB   ← convert before passing in, convert after receiving
Gradio       →  RGB   ← convert before passing in, convert after receiving
PIL          →  RGB   ← convert before passing in, convert after receiving
Skin work    →  LAB   ← convert to LAB, operate, convert back to BGR
```

No implicit assumptions about what the caller passed. Missing `cvtColor` at a boundary is an automatic CRITICAL finding.

---

## 3. Mask Precision Contract

```
MASK CONTRACT {
  dtype:       float32
  range:       [0.0, 1.0]          ← NOT [0, 255]
  feathering:  gaussian blur BEFORE thresholding, not after
  forbidden:   binary masks for skin region blending
  forbidden:   uint8 mask multiplication
}
```

BiSeNet output is a probability map. Use it as a soft blend weight. Converting to uint8 and multiplying loses sub-pixel precision at mask edges and produces visible hard boundaries in skin compositing.

**Feathering order matters**: Blurring a hard binary mask produces a different (worse) result than blurring before thresholding. Always blur the probability map, then threshold if a hard decision is needed.

**Resampling rules**:
- Downsampling masks: `cv2.INTER_AREA`
- Upsampling masks: `cv2.INTER_LINEAR`
- Never `cv2.INTER_NEAREST` for skin masks (produces staircase artifacts at edges)

---

## 4. Kernel Sizing Contract

Hardcoded pixel radii are wrong at multiple resolutions.

```python
# Forbidden
kernel_size = 5  # works at 800px, invisible at 4K

# Correct
kernel_size = max(3, int(image_width * KERNEL_SCALE))
kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # must be odd
```

Define `KERNEL_SCALE` constants in `params.py`. Never hardcode in the processing function.

---

## 5. Hallucination Traps

Common wrong-but-compiles patterns. All are automatic CRITICAL or MAJOR findings in the Review Loop.

**General traps:**
- Referencing PHP, Laravel, Yii2 (this is a Python project).
- Assuming CUDA/CoreML is available. Always provide CPU fallback.
- Using deprecated helpers (`combine_adaptive`, `SkinProcessor.smooth()`).

**Domain traps:**

| Trap | Wrong | Correct |
|------|-------|---------|
| Skin smoothing | `cv2.GaussianBlur(skin_region, ...)` | Frequency separation: process low-freq band, preserve high-freq. Never blur texture directly. |
| BiSeNet mask usage | `mask = (bisenet_output > 0.5).astype(np.uint8)` | Use as float32 blend weight; soft composite |
| uint8 arithmetic | `result = img_uint8 + adjustment` | Convert to float32 first |
| Skin tone in BGR | Operate on B, G, R channels for skin equalization | Convert to LAB; operate on L and a,b channels separately |
| Hardcoded kernels | `cv2.GaussianBlur(img, (5, 5), 0)` | Compute size from image dimensions via KERNEL_SCALE |
| Feathering order | `blur(threshold(mask))` | `threshold(blur(mask))` |
| Mask downsampling | `cv2.resize(mask, ..., cv2.INTER_LINEAR)` | Use `cv2.INTER_AREA` for downsampling |
| Style before features | Running `style.py` before `lips.py` / `eyes.py` | Follow pipeline order: features → style → grading |

---

## 6. Precision Audit Checklist

Run before finalizing any implementation that touches pixel arrays.

- [ ] All intermediate arrays are `float32`
- [ ] No `np.uint8` or `np.int8` arithmetic between pipeline stages
- [ ] Input converted on first read: `img.astype(np.float32) / 255.0`
- [ ] Output converted at last write only: `np.clip(result * 255, 0, 255).astype(np.uint8)`
- [ ] Every image function has colorspace declared in its docstring
- [ ] All cross-boundary colorspace conversions use explicit `cv2.cvtColor`
- [ ] BiSeNet masks are float32 in [0.0, 1.0] before compositing
- [ ] Mask feathering applied before thresholding, not after
- [ ] Kernel sizes computed from image dimensions via KERNEL_SCALE, not hardcoded
- [ ] Mask downsampling uses `cv2.INTER_AREA`; upsampling uses `cv2.INTER_LINEAR`

If any item is unchecked, the implementation is not ready for review.