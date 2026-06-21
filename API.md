# Pro Max Face Retouch Engine — API Reference

This document provides a comprehensive API reference for the `retouch` package. It covers the core orchestrator `RetouchEngine`, the convenience wrapper function `retouch`, structured return types, parameter contexts, and style library helpers.

---

## 1. Package Entry Point

You can import the main engine orchestrator and the one-shot convenience function directly from the root of the package:

```python
from retouch import RetouchEngine, retouch, __version__
```

---

## 2. Core Orchestrator: `RetouchEngine`

The `RetouchEngine` class is the main orchestrator of the retouching pipeline. It initializes the face detector, facial landmarker, semantic parser, and specialized processors (skin smoothing, relighting, makeup, etc.).

### Lifecycle & Context Manager

The engine manages underlying MediaPipe resources and supports explicit resource cleanup. It can be used as a context manager to ensure release of native resources:

```python
# Recommended: Context manager usage
with RetouchEngine(max_faces=5) as engine:
    result = engine.process(img_bgr, recipe="cosplay")

# Manual lifecycle management
engine = RetouchEngine(max_faces=5)
try:
    result = engine.process(img_bgr, recipe="cosplay")
finally:
    engine.close()
```

### `__init__` Parameters

*   **`max_faces`** (`int`, default: `10`): Maximum number of faces to detect and process concurrently.
*   **`min_confidence`** (`float`, default: `0.5`): Minimum detection confidence score for face landmarker.

---

### `process()` Method

Run the full face retouching pipeline on a single image.

```python
def process(
    self,
    img_bgr: np.ndarray,
    recipe: Optional[str] = None,
    preset: Optional[str] = None,
    # Overrides (None -> fallback to recipe defaults)
    ...
    fast: bool = False,
    style_profile: Optional[StyleProfile] = None,
    style_ref: Optional[np.ndarray] = None,
    debug_dir: Optional[str] = None,
) -> ProcessingResult:
```

#### Inputs
*   **`img_bgr`** (`np.ndarray`): Input portrait image, expected in standard BGR format (e.g. from `cv2.imread`).
*   **`recipe` / `preset`** (`str`, optional): Built-in preset recipe key (e.g. `"natural"`, `"cosplay"`, `"cyber_doll"`, `"portrait"`, `"pink_dream"`, `"meitu_clone"`). Preset is a backward-compatible alias.
*   **`fast`** (`bool`, default: `False`): Fast-preview path. Downsamples the input image to a maximum dimension of 800px *before* landmarker and segmenter runs, saving ~40% overhead on high-res photos.
*   **`style_profile`** (`StyleProfile`, optional): Active custom style overrides loaded from json/library.
*   **`style_ref`** (`np.ndarray`, optional): Reference image used to extract style dynamics on-the-fly.
*   **`debug_dir`** (`str`, optional): Directory path where intermediate masks and stages will be written for debugging.

#### Parameter Overrides (grouped logically)
Passing an explicit value override to these parameters takes precedence over the active recipe's defaults. Pass `None` to fallback.

##### **Skin Smoothing & Texture**
*   **`smooth`** (`float`, `0` to `100`): Face smoothing amount (median/blur blend strength).
*   **`nose_smooth`** (`float`, `0` to `100`, optional): Additional smoothing on the nose bridge specifically. If `None` or `< 0`, follows face default.
*   **`mid_reduction`** (`float`, `0.0` to `1.0`): Suppression amount of mid-frequency details (blemishes, redness) while keeping pores intact.
*   **`texture_opacity`** (`float`, `0.0` to `1.0`): Visibility opacity of high-frequency skin pore detail overlays.
*   **`pore_synthesis`** (`float`, `0` to `100`): Added micro-pores amount to prevent an artificial plastic skin look.
*   **`blemish`** (`float`, `0` to `100`): Blemish/spot inpainting strength.

##### **Skin Tone & Color**
*   **`whiten`** (`float`, `0` to `100`): Foundation luminance boost.
*   **`whiten_tone`** (`str`, `"rosy"`, `"porcelain"`, or `"neutral"`): Foundation color target.
*   **`equalize`** (`float`, `0` to `100`): CLAHE-based skin tone leveling and redness normalization.
*   **`white_costume_lift`** (`bool`): Boost luminance of white clothing/costumes to create depth.
*   **`dodge_burn`** (`float`, `0` to `100`): Face structure sculpting via manual or automatic highlight/shadow maps.

##### **3D Relighting**
*   **`relight`** (`float`, `0` to `100`): Strength of 3D virtual studio light source direction adjustment.
*   **`relight_azimuth`** (`float`, `-180` to `180`): Horizontal light source direction angle.
*   **`relight_elevation`** (`float`, `-90` to `90`): Vertical light source elevation angle.

##### **Eyes & Lips**
*   **`eye_enhance`** (`float`, `0` to `100`): Boost iris reflection contrast, clarity, and sclera brightness.
*   **`dark_circles`** (`float`, `0` to `100`): Under-eye bag dark circles reduction.
*   **`teeth_whiten`** (`float`, `0` to `100`): Whiten and brighten teeth enamel.
*   **`lip_enhance`** (`float`, `0` to `100`): Enhance lip texture definition, gloss, and contour.
*   **`lip_tint`** (`str` or `None`, e.g. `"cosplay"`, `"rose"`, `"pink"`, `"coral"`, `"natural"`, `"berry"`): Apply a natural cosmetic color tone to the lips.
*   **`lip_finish`** (`str`, default: `"gloss"`): Set texture finish for lips.

##### **Makeup & Face Structure**
*   **`blush`** (`float`, `0` to `100`): Cheeks blush makeup strength.
*   **`nose_blush`** (`bool`): Apply subtle blush to the tip of the nose (common in anime/cosplay presets).
*   **`under_eye_blush`** (`bool`): Apply soft blush around the under-eye area.
*   **`slimming`** (`float`, `0` to `100`): Liquid face reshaping (jawline and cheek slimming).

##### **Hair & Subject separation**
*   **`hair_enhance`** (`float`, `0` to `100`): Boost highlight gloss reflections in hair strands.
*   **`subject_separation`** (`float`, `0` to `100`): Boost background contrast and light separation using selfie segmentation.

##### **Global Tonal Adjustments**
*   **`auto_exposure`** (`bool`, default: `False`): Automatically balance face-region exposure before pipeline execution.
*   **`brightness`** (`float`, `-100` to `100`): Tonal brightness offset.
*   **`contrast`** (`float`, `-50` to `50`): Tonal contrast adjustment.
*   **`highlights`** (`float`, `-100` to `100`): Recover or boost highlight regions.
*   **`shadows`** (`float`, `-100` to `100`): Adjust shadow regions.
*   **`whites`** (`float`, `-100` to `100`): White point ceiling adjustment.
*   **`blacks`** (`float`, `-100` to `100`): Black point floor adjustment.
*   **`clarity`** (`float`, `0` to `100`): Local mid-frequency contrast enhancement.
*   **`vibrance`** (`float`, `0` to `100`): Adjust non-skin/under-saturated color vibrance.
*   **`saturation`** (`float`, `-100` to `100`): Global saturation adjustment.

##### **Color Grading & Tone Mapping**
*   **`color_grade`** (`str`): Color grade preset name (e.g. `"kodak_gold"`, `"fuji_astia"`, `"nordic_frost"`, `"retro_film"`).
*   **`grade_intensity`** (`float`, `0.0` to `1.0`): Opacity/intensity of the selected color grade preset.
*   **`lut`** (`str`): Absolute filepath to a `.cube` or `.png` 3D Lookup Table file.
*   **`color_grade_stack`** (`list` of `dict`, optional): List of dictionary objects to composite multiple color presets sequentially.
*   **`color_ref`** (`np.ndarray`): Reference image array for matching color tone.
*   **`color_transfer_intensity`** (`float`, `0.0` to `1.0`): Mix ratio of color transfer from `color_ref`.

##### **Screen Effects & Finish**
*   **`bloom`** (`float`, `0` to `100`): Orton screen glow opacity.
*   **`bloom_threshold`** (`float`, `150` to `250`): Luminance starting threshold for bloom bleed.
*   **`bloom_softness`** (`float`, `1` to `100`): Bloom blur radius.
*   **`specular_bloom`** (`float`, `0` to `100`): Soft glow applied specifically to skin highlight zones.
*   **`specular_bloom_tone`** (`str`, `"rosy"` or `"porcelain"`): Specular bloom tint tone.
*   **`chromatic_aberration`** (`float`): Lens chromatic aberration displacement amount.
*   **`halation`** (`float`): Film halation glow strength.
*   **`grain`** (`float`): Film grain noise opacity.
*   **`vignette`** (`float`, `0` to `100`): Radial lens darkening amount.
*   **`sharpen`** (`float`): Final unsharp mask sharpening amount.
*   **`sharpen_radius`** (`float`): Radius of the sharpening filter.
*   **`impact`** (`float`, `0` to `100`): Dynamic contrast finish impact parameter.
*   **`glow`** (`float`): Soft dreamy glow effect strength.

---

## 3. Convenience Function: `retouch()`

For simple scripts or backward-compatibility, you can use the `retouch` module-level wrapper. It handles the instantiation and teardown of the engine automatically.

```python
def retouch(
    img_bgr: np.ndarray,
    smooth: float = 50,
    whiten: float = 30,
    eye_enhance: float = 30,
    contrast: float = 0,
    preset: Optional[str] = None,
    **kwargs,
) -> ProcessingResult:
```

### Usage
```python
import cv2
from retouch import retouch

img = cv2.imread("face.jpg")
# Automatically spins up and shuts down a RetouchEngine instance
out = retouch(img, preset="cosplay", smooth=65, whiten=15)
cv2.imwrite("output.jpg", out)
```

---

## 4. Return Object: `ProcessingResult`

The return value of both `RetouchEngine.process()` and `retouch()` is a rich `ProcessingResult` object. 

### Legacy-Compatible Array Representation
`ProcessingResult` inherits from `numpy.ndarray` and points to the processed BGR image array. This guarantees that **any legacy code, OpenCV methods, or visualization libraries that expect a standard numpy image array will accept it immediately** without modification.

```python
result = engine.process(img)

# Directly access shape, slices, and dtype like a normal numpy array
h, w, c = result.shape
crop = result[100:200, 100:200]
print(result.dtype)  # dtype('uint8')
```

### Extended Debug & Performance Metadata

*   **`result.image`** (`np.ndarray`): The raw output BGR image array.
*   **`result.face_count`** (`int`): Count of unique faces detected and processed.
*   **`result.skin_mask`** (`np.ndarray`): A normalized single-channel float32 mask `[0.0, 1.0]` of the processed skin regions.
*   **`result.skin_hair_mask`** (`np.ndarray`): A combined float32 mask of the skin and hair regions.
*   **`result.lips_mask`** (`np.ndarray`): A mask of the lips region.
*   **`result.sharpen_mask`** (`np.ndarray`): The mask applied during the final sharpening stage.
*   **`result.params`** (`ProcessingContext`): The fully-resolved `ProcessingContext` containing all parameters (recipe values + manual overrides) applied.
*   **`result.timings`** (`Dict[str, float]`): Timing metrics (in milliseconds) for each stage of the pipeline. Example:
    ```python
    print(result.timings)
    # Output:
    # {
    #   'detection': 14.5,
    #   'reshape': 8.2,
    #   'per_face': 120.3,
    #   'global': 5.4,
    #   'grading': 24.1
    # }
    ```

---

## 5. Style Library & Machine Learning APIs

The `retouch` package provides tools to extract editing patterns from Lightroom/Photoshop-edited image pairs and save them as reusable custom styles.

### `StyleProfile` Dataclass

Represents the mathematical delta between an original unedited photo and an edited photo.

```python
@dataclass
class StyleProfile:
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    saturation_delta: float = 0.0
    skin_l_mean_delta: float = 0.0
    skin_a_mean_delta: float = 0.0
    skin_b_mean_delta: float = 0.0
    skin_smooth_strength: float = 0.0
    skin_mid_reduction: float = 0.0
    skin_texture_opacity: float = 1.0
```

#### Serialization Methods
*   **`to_dict()`** -> `Dict[str, Any]`
*   **`to_json()`** -> `str`
*   **`save(filepath: str)`**: Save the profile to a JSON file.
*   **`from_dict(d: Dict[str, Any])`** -> `StyleProfile`
*   **`from_json(s: str)`** -> `StyleProfile`
*   **`load(filepath: str)`** -> `StyleProfile`

---

### Style Library Functions

`from retouch.style_library import list_styles, save_style_profile, learn_dataset_style`

#### `list_styles`
```python
def list_styles(directory: Path | str = DEFAULT_STYLE_DIR) -> List[Dict[str, Any]]
```
Lists all custom style profiles in the JSON library with their metadata fields (e.g. `name`, `author`, `tags`, `profile`).

#### `save_style_profile`
```python
def save_style_profile(
    name: str,
    profile: StyleProfile | Dict[str, float],
    author: Optional[str] = None,
    version: str = "1.0",
    tags: Optional[List[str]] = None,
    directory: Path | str = DEFAULT_STYLE_DIR,
) -> Path:
```
Wraps and saves a profile with creation dates and version numbering. Protects history by auto-incrementing suffixes (e.g., `_v2`) if the name already exists.

#### `learn_dataset_style`
```python
def learn_dataset_style(
    original_dir: str | Path,
    edited_dir: str | Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[StyleProfile, int]:
```
Performs machine learning over a folder pair. It matches image filenames, filters invalid outliers using a trimmed mean, ignores failed face detections, and averages color and skin texture parameters. Returns the final averaged `StyleProfile` and the count of successfully processed image pairs.
