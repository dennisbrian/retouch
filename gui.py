#!/usr/bin/env python3

import base64
import logging
import sys
import os
import shutil
import time
import tempfile
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine
from retouch.engine import resolve_recipe
from retouch.io import imread_exif, EXPORT_RES_MAP, EXT_MAP
from retouch.lips import LIP_TINT_NAMES
from retouch.recipes import RECIPES
from retouch.params import recipe_to_params, PROCESSING_PARAMS, param_names, gui_values_to_engine_kwargs
from retouch.grading import list_available_presets
from retouch.style_library import list_styles, save_style_profile, learn_dataset_style
from retouch.batch_processor import BatchProcessor
from retouch.style import StyleProfile

_logger = logging.getLogger(__name__)

RECIPE_NAMES = list(RECIPES.keys())
COLOR_GRADE_NAMES = ["none"] + list_available_presets()
LUT_CHOICES = ["none", "kodak", "fuji"]
_TONE_CHOICES = ["rosy", "porcelain", "neutral"]
WHITEN_TONE_CHOICES = _TONE_CHOICES
LIP_FINISH_CHOICES = ["gloss", "matte", "velvet"]
SPECULAR_BLOOM_TONE_CHOICES = _TONE_CHOICES

PREVIEW_MAX_HEIGHT = 900
COMPARE_SEPARATOR_WIDTH = 4
COMPARE_SEPARATOR_COLOR = 200
TEMP_CLEANUP_AGE_SEC = 300

_engine = None
_engine_lock = threading.Lock()
def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RetouchEngine()
    return _engine


def recipe_defaults(recipe_name):
    """Return the per-slider defaults for a recipe.

    Delegates to ``retouch.params.recipe_to_params`` so there is exactly
    one place that knows how to turn a recipe dict into a UI-side values
    bag.  The engine uses the same spec list from a different code path
    (``engine.build_context``).
    """
    return recipe_to_params(recipe_name)


def get_custom_style_names():
    styles = list_styles()
    return [s["name"] for s in styles]


def apply_custom_style(style_name, current_recipe="natural"):
    if not style_name:
        return tuple([gr.update()] * len(_recipe_outputs))

    styles = list_styles()
    target = None
    for s in styles:
        if s["name"] == style_name:
            target = s
            break

    if not target:
        return tuple([gr.update()] * len(_recipe_outputs))
        
    p_dict = target["profile"]
    profile = StyleProfile(**p_dict)
    
    d = recipe_defaults(current_recipe)
    
    d["smooth"] = int(np.clip(profile.skin_smooth_strength * 100.0, 0.0, 100.0))
    d["whiten"] = int(np.clip(profile.skin_l_mean_delta * 4.0, 0.0, 100.0))
    d["mid_reduction"] = float(np.clip(profile.skin_mid_reduction, 0.0, 1.0))
    d["texture_opacity"] = float(np.clip(profile.skin_texture_opacity, 0.0, 1.0))
    d["contrast"] = int(profile.contrast_delta)
    d["brightness"] = int(np.clip(profile.brightness_delta, -50.0, 50.0))
    
    return (
        d["smooth"], d["mid_reduction"], d["texture_opacity"], d["pore_synthesis"], d["nose_smooth"], d["micro_restore"],
        d["whiten"], d["equalize"], d["blemish"], d["whiten_tone"], d["nose_blush"], d["under_eye_blush"], d["white_costume_lift"],
        d["dodge_burn"], d["relight"], d["relight_azimuth"], d["relight_elevation"], d["specular_bloom"], d["specular_bloom_tone"],
        d["eye_enhance"], d["catchlight"], d["dark_circles"], d["teeth_whiten"], d["lip_enhance"], d["lip_tint"], d["lip_finish"], d["blush"], d["slimming"], d["hair_enhance"],
        d["contrast"], d["brightness"], d["highlights"], d["shadows"], d["whites"], d["blacks"], d["clarity"], d["vibrance"], d["saturation"], d["auto_exposure"],
        d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["glow"], d["vignette"], d["sharpen"], d["sharpen_radius"], d["subject_separation"], d["impact"],
        d["color_grade"], d["grade_intensity"],
        d["chromatic_aberration"], d["grain"], d["halation"], d["lut"],
        d["tonal_curve_strength"], d["skin_protect_strength"], d["grain_strength"], d["highlight_rolloff_strength"],
        d["shadow_hue"], d["shadow_sat"], d["midtone_hue"], d["midtone_sat"], d["highlight_hue"], d["highlight_sat"],
    )


def on_save_style(style_name, author, tags_str,
                  smooth, mid_reduction, texture_opacity,
                  whiten, contrast, brightness):
    style_name = style_name.strip()
    if not style_name:
        return gr.update(), gr.update(), "Error: Style name cannot be empty."
    if len(style_name) > 100:
        return gr.update(), gr.update(), "Error: Style name must be 100 characters or less."

    tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    profile = StyleProfile(
        brightness_delta=float(brightness / 2.0),
        contrast_delta=float(contrast),
        saturation_delta=0.0,
        skin_l_mean_delta=float(whiten / 4.0),
        skin_a_mean_delta=0.0,
        skin_b_mean_delta=0.0,
        skin_smooth_strength=float(smooth / 100.0),
        skin_mid_reduction=float(mid_reduction),
        skin_texture_opacity=float(texture_opacity),
    )

    try:
        save_style_profile(
            name=style_name,
            profile=profile,
            author=author or "Dennis",
            tags=tags,
        )
        choices = get_custom_style_names()
        gr.Info(f"Style '{style_name}' saved!")
        return gr.update(choices=choices, value=style_name), gr.update(choices=choices, value=style_name), f"Style '{style_name}' saved successfully!"
    except Exception as e:
        return gr.update(), gr.update(), f"Failed to save style: {e}"


def on_learn_style(orig_dir, edit_dir, style_name, author, tags_str, prg=gr.Progress()):
    if not orig_dir or not edit_dir:
        return gr.update(), gr.update(), "Error: Original and Edited folders must be specified."
    style_name = style_name.strip()
    if not style_name:
        return gr.update(), gr.update(), "Error: Style name cannot be empty."
    if len(style_name) > 100:
        return gr.update(), gr.update(), "Error: Style name must be 100 characters or less."

    def prg_cb(progress, message):
        prg(progress, desc=message)

    try:
        profile, count = learn_dataset_style(orig_dir, edit_dir, progress_callback=prg_cb)
        if count == 0:
            return gr.update(), gr.update(), "No matching image pairs were found or analyzed successfully."

        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        save_style_profile(
            name=style_name,
            profile=profile,
            author=author or "Dennis",
            tags=tags,
        )
        
        choices = get_custom_style_names()
        gr.Info(f"Learned style '{style_name}' from {count} pairs!")
        return gr.update(choices=choices, value=style_name), gr.update(choices=choices, value=style_name), f"Extracted & saved style '{style_name}' from {count} image pairs!"
    except Exception as e:
        return gr.update(), gr.update(), f"Error during dataset learning: {e}"


def on_process_folder(input_dir, output_dir, style_type, custom_style_name, recipe_name,
                      export_fmt, export_quality, export_res, auto_group, generate_sheet, export_zip,
                      prg=gr.Progress()):
    if not input_dir or not output_dir:
        return None, None, "Error: Both Input and Output directories must be specified."

    gr.Info("Batch processing started...")
    processor = BatchProcessor()
    
    profile = None
    style_val = recipe_name
    
    if style_type == "Use Custom Style":
        style_val = ""
        if not custom_style_name:
            return None, None, "Error: Please select a custom style profile."
        styles = list_styles()
        for s in styles:
            if s["name"] == custom_style_name:
                profile = StyleProfile(**s["profile"])
                break
        if profile is None:
            return None, None, f"Error: Custom style '{custom_style_name}' not found."

    def prg_cb(progress, message):
        print(f"[Batch Progress {progress * 100:.1f}%]: {message}")
        prg(progress, desc=message)

    try:
        processed, sheet_path, zip_path, log = processor.process_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            style_name_or_recipe=style_val,
            custom_style_profile=profile,
            export_fmt=export_fmt,
            export_quality=export_quality,
            export_res=export_res,
            auto_group=auto_group,
            generate_sheet=generate_sheet,
            export_zip=export_zip,
            progress_callback=prg_cb,
        )
        
        gr.Info("Batch processing complete!")
        return sheet_path, zip_path, log
    except Exception as e:
        gr.Warning(f"Batch processing failed: {e}")
        return None, None, f"Exception during batch processing: {e}"


# PROCESS_INPUT_KEYS — the ordered list of inputs the process_image() Gradio
# event handler expects.  Generated from PROCESSING_PARAMS (in declaration
# order) so the slider order stays in lock-step with the spec, plus the
# fixed-prefix transport / session keys at the end.
PROCESS_INPUT_KEYS = (
    ["img_paths", "recipe"]
    + param_names()
    + [
        "color_ref_img", "color_ref_strength",
        "show_compare", "fast",
        "export_fmt", "export_quality", "export_res",
        "debug_mode",
    ]
)

def _coerce_float(value: object, default: float = 0.0) -> float:
    """Safely coerce an arbitrary Gradio value to float, falling back on parse failure.

    Slider components always return ``float``, but transient states (watchfiles
    reload races, empty Form payloads, deprecated field names mapped onto the
    same key) can deliver strings or ``None`` here.  Prefer the numeric value
    if one can be parsed; otherwise fall back to *default* rather than 500-ing
    the whole request.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    if value is None:
        return default
    return default


def process_image(*args):
    params = dict(zip(PROCESS_INPUT_KEYS, args))
    img_paths = params.get("img_paths")
    recipe = params.get("recipe")
    color_ref_img = params.get("color_ref_img")
    color_ref_strength = _coerce_float(params.get("color_ref_strength"))
    show_compare = params.get("show_compare")
    fast = params.get("fast")
    export_fmt = params.get("export_fmt")
    export_quality = params.get("export_quality")
    export_res = params.get("export_res")
    debug_mode = params.get("debug_mode")

    if not img_paths:
        return None, gr.update(visible=False), None, None, "Please upload an image first.", None, gr.update(visible=False)

    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    gr.Info(f"Processing {len(img_paths)} image(s)...")

    exported_paths = []
    first_result_rgb = None
    first_combined = None
    first_original = None
    first_result = None
    debug_images = []

    color_ref_bgr = None
    if color_ref_img is not None and color_ref_strength > 0:
        if isinstance(color_ref_img, dict):
            color_ref_img = color_ref_img.get("name") or color_ref_img.get("path")
        color_ref_bgr = imread_exif(color_ref_img)

    engine = get_engine()
    start = time.time()

    # Clean up older temp directories and ZIPs from previous runs (older than 5 minutes)
    try:
        temp_root = Path(tempfile.gettempdir())
        now = time.time()
        for p in temp_root.glob("retouch_tmp_*"):
            if p.is_dir() and (now - p.stat().st_mtime > TEMP_CLEANUP_AGE_SEC):
                shutil.rmtree(p, ignore_errors=True)
        for p in temp_root.glob("retouch_export_*.zip"):
            if p.is_file() and (now - p.stat().st_mtime > TEMP_CLEANUP_AGE_SEC):
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception as e:
        _logger.warning("Temp directory cleanup warning: %s", e)

    temp_dir = tempfile.mkdtemp(prefix="retouch_tmp_")
    debug_dir = os.path.join(temp_dir, "debug") if debug_mode else None

    # Translate the GUI-side values dict into the engine-side kwargs dict.
    # The spec list (in retouch.params) is the source of truth for the
    # unit conversions (e.g. grain × 500 → grain / 500, "none" → None,
    # grade_intensity / 100, etc.).
    engine_kwargs = gui_values_to_engine_kwargs(
        params,
        extra={
            "recipe": recipe,
            "color_ref": color_ref_bgr,
            "color_transfer_intensity": color_ref_strength,
            "fast": fast,
            "debug_dir": None,  # set per-image below
        },
    )

    for idx, path_item in enumerate(img_paths):
        try:
            curr_path = path_item
            if isinstance(path_item, dict):
                curr_path = path_item.get("name") or path_item.get("path")
            
            img_bgr = imread_exif(curr_path)
            original = img_bgr.copy()

            engine_kwargs["debug_dir"] = (
                debug_dir if (first_result_rgb is None and first_combined is None) else None
            )

            result = engine.process(img_bgr, **engine_kwargs)

            if first_result_rgb is None and first_combined is None:
                first_original = original
                first_result = result
                first_result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                if show_compare:
                    h = min(original.shape[0], result.shape[0])
                    sep = np.full((h, COMPARE_SEPARATOR_WIDTH, 3), COMPARE_SEPARATOR_COLOR, dtype=np.uint8)
                    orig_rgb = cv2.cvtColor(original[:h], cv2.COLOR_BGR2RGB)
                    res_rgb = first_result_rgb[:h]
                    combined = np.hstack([orig_rgb, sep, res_rgb])
                    max_h = PREVIEW_MAX_HEIGHT
                    if combined.shape[0] > max_h:
                        scale = max_h / combined.shape[0]
                        new_w = int(combined.shape[1] * scale)
                        combined = cv2.resize(combined, (new_w, max_h), interpolation=cv2.INTER_AREA)
                    first_combined = combined
                else:
                    if first_result_rgb.shape[0] > PREVIEW_MAX_HEIGHT:
                        scale = PREVIEW_MAX_HEIGHT / first_result_rgb.shape[0]
                        new_w = int(first_result_rgb.shape[1] * scale)
                        first_result_rgb = cv2.resize(first_result_rgb, (new_w, PREVIEW_MAX_HEIGHT), interpolation=cv2.INTER_AREA)

                if debug_mode and debug_dir and os.path.isdir(debug_dir):
                    mask_files = [
                        ("Skin Mask", "skin_mask.png"),
                        ("Skin+Hair Mask", "skin_hair_mask.png"),
                        ("Lips Mask", "lips_mask.png"),
                        ("Sharpen Mask", "sharpen_mask.png"),
                        ("Glow Mask", "glow_mask.png"),
                        ("Freq Low", "freq_low.png"),
                        ("Freq Mid", "freq_mid.png"),
                        ("Freq High", "freq_high.png"),
                    ]
                    for label, fname in mask_files:
                        mpath = os.path.join(debug_dir, fname)
                        if os.path.exists(mpath):
                            mask_img = cv2.imread(mpath)
                            if mask_img is not None:
                                debug_images.append((cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB), label))

            export_img = result
            export_max = EXPORT_RES_MAP.get(export_res)
            if export_max is not None:
                h, w = export_img.shape[:2]
                if max(h, w) > export_max:
                    scale = export_max / max(h, w)
                    export_img = cv2.resize(export_img, (int(w * scale), int(h * scale)),
                                            interpolation=cv2.INTER_AREA)

            ext = EXT_MAP.get(export_fmt, ".jpg")
            filename = Path(curr_path).stem
            out_path = os.path.join(temp_dir, f"{filename}_{idx:03d}_retouched{ext}")
            write_params = []
            if export_fmt == "JPEG":
                write_params = [cv2.IMWRITE_JPEG_QUALITY, export_quality]
            elif export_fmt == "WebP":
                write_params = [cv2.IMWRITE_WEBP_QUALITY, export_quality]
            cv2.imwrite(out_path, export_img, write_params)
            exported_paths.append(out_path)

        except Exception as e:
            _logger.exception("Failed to process %s", path_item)
            from retouch.utils import log_crash
            crash_path = log_crash(e, {
                "recipe": recipe,
                "image_path": str(curr_path),
                "show_compare": show_compare,
                "fast": fast
            })
            if crash_path:
                _logger.info("Crash details saved to: %s", crash_path)

    if not exported_paths:
        gr.Warning("No images were successfully processed.")
        return None, gr.update(visible=False), None, None, "Error: No images were successfully processed.", None, gr.update(visible=False)

    preview = first_combined if show_compare else first_result_rgb
    debug_gallery = debug_images if debug_images else None
    debug_vis = gr.update(visible=bool(debug_images))

    elapsed = time.time() - start
    slide_html = _make_comparison_html(first_original, first_result) if (first_original is not None and first_result is not None) else ""
    if len(exported_paths) > 1:
        zip_stamp = time.strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(tempfile.gettempdir(), f"retouch_export_{zip_stamp}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for exp_path in exported_paths:
                zipf.write(exp_path, arcname=os.path.basename(exp_path))
        gr.Info(f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s")

        if show_compare:
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_original, zip_path, f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s ✓", debug_gallery, debug_vis
        return preview, gr.update(visible=False), first_original, zip_path, f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s ✓", debug_gallery, debug_vis
    else:
        gr.Info(f"Done in {elapsed:.1f}s")

        if show_compare:
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_original, exported_paths[0], f"Done in {elapsed:.1f}s ✓", debug_gallery, debug_vis
        return preview, gr.update(visible=False), first_original, exported_paths[0], f"Done in {elapsed:.1f}s ✓", debug_gallery, debug_vis


def on_recipe_change(recipe):
    d = recipe_defaults(recipe)
    return (
        d["smooth"], d["mid_reduction"], d["texture_opacity"], d["pore_synthesis"], d["nose_smooth"], d["micro_restore"],
        d["whiten"], d["equalize"], d["blemish"], d["whiten_tone"], d["nose_blush"], d["under_eye_blush"], d["white_costume_lift"],
        d["dodge_burn"], d["relight"], d["relight_azimuth"], d["relight_elevation"], d["specular_bloom"], d["specular_bloom_tone"],
        d["eye_enhance"], d["catchlight"], d["dark_circles"], d["teeth_whiten"], d["lip_enhance"], d["lip_tint"], d["lip_finish"], d["blush"], d["slimming"], d["hair_enhance"],
        d["contrast"], d["brightness"], d["highlights"], d["shadows"], d["whites"], d["blacks"], d["clarity"], d["vibrance"], d["saturation"], d["auto_exposure"],
        d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["glow"], d["vignette"], d["sharpen"], d["sharpen_radius"], d["subject_separation"], d["impact"],
        d["color_grade"], d["grade_intensity"],
        d["chromatic_aberration"], d["grain"], d["halation"], d["lut"],
        d["tonal_curve_strength"], d["skin_protect_strength"], d["grain_strength"], d["highlight_rolloff_strength"],
        d["shadow_hue"], d["shadow_sat"], d["midtone_hue"], d["midtone_sat"], d["highlight_hue"], d["highlight_sat"],
    )


def reset_skin_smoothing(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["smooth"], d["nose_smooth"], d["mid_reduction"], d["texture_opacity"], d["micro_restore"], d["pore_synthesis"], d["blemish"]

def reset_skin_tone(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["whiten"], d["whiten_tone"], d["equalize"], d["auto_exposure"], d["white_costume_lift"]

def reset_basic_tone(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["contrast"], d["brightness"], d["clarity"], d["vibrance"], d["saturation"]

def reset_tone_curve(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["highlights"], d["shadows"], d["whites"], d["blacks"]

def reset_relighting(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["relight"], d["relight_azimuth"], d["relight_elevation"]

def reset_eyes_lips(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["eye_enhance"], d["catchlight"], d["dark_circles"], d["teeth_whiten"], d["lip_enhance"], d["lip_tint"], d["lip_finish"], d["blush"], d["nose_blush"], d["under_eye_blush"]

def reset_face_reshaping(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["slimming"]

def reset_structure_effects(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["hair_enhance"], d["dodge_burn"], d["impact"], d["specular_bloom"], d["specular_bloom_tone"], d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["sharpen"], d["sharpen_radius"], d["glow"], d["vignette"], d["subject_separation"]

def reset_color_grading(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["color_grade"], d["grade_intensity"]

def reset_film_effects(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["chromatic_aberration"], d["grain"], d["halation"], d["lut"], d["tonal_curve_strength"], d["skin_protect_strength"], d["grain_strength"], d["highlight_rolloff_strength"]

def reset_split_toning(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["shadow_hue"], d["shadow_sat"], d["midtone_hue"], d["midtone_sat"], d["highlight_hue"], d["highlight_sat"]

def reset_color_transfer():
    return None, 1.0

def reset_debug(recipe_name):
    return False



LIP_TINTS = ["none"] + LIP_TINT_NAMES
custom_style_choices = get_custom_style_names()

COMPARE_TPL = """
<div id="cmp-%(uid)s" style="position:relative;width:100%%;user-select:none;overflow:hidden;border-radius:4px">
  <img src="%(result)s" style="width:100%%;display:block;pointer-events:none">
  <div class="cmp-overlay" style="position:absolute;top:0;left:0;width:50%%;height:100%%;overflow:hidden">
    <img src="%(orig)s" style="width:100%%;display:block;max-width:none;position:absolute;left:0;top:0;pointer-events:none">
  </div>
  <div class="cmp-handle" style="position:absolute;top:0;left:50%%;width:3px;height:100%%;background:#fff;cursor:ew-resize;z-index:10;box-shadow:0 0 6px rgba(0,0,0,0.4)"></div>
  <div class="cmp-label" style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.55);color:#fff;padding:2px 10px;border-radius:3px;font-size:11px;letter-spacing:1px;pointer-events:none">BEFORE</div>
  <div class="cmp-label" style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.55);color:#fff;padding:2px 10px;border-radius:3px;font-size:11px;letter-spacing:1px;pointer-events:none">AFTER</div>
</div>
"""


def _make_comparison_html(orig_bgr, result_bgr, max_height=600):
    scale = max_height / max(orig_bgr.shape[0], result_bgr.shape[0])
    if scale < 1.0:
        new_w = int(orig_bgr.shape[1] * scale)
        new_h = int(orig_bgr.shape[0] * scale)
        orig_bgr = cv2.resize(orig_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result_bgr = cv2.resize(result_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, ob = cv2.imencode('.jpg', orig_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    _, rb = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    uid = hex(int(time.time() * 1e6))[2:]
    return COMPARE_TPL % {
        "uid": uid,
        "orig": f"data:image/jpeg;base64,{base64.b64encode(ob).decode()}",
        "result": f"data:image/jpeg;base64,{base64.b64encode(rb).decode()}",
    }

with gr.Blocks(title="🪄 Retouch — AI Portrait Workflow Platform", theme=gr.themes.Soft(primary_hue="sky", secondary_hue="slate"), css="""
    /* --- Liquid Glass Theme --- */
    
    /* Scoped under .dark to avoid light mode text visibility issues */
    .dark.gradio-container, .dark .gradio-container {
        --background-fill-primary: transparent !important;
        --background-fill-primary-dark: transparent !important;
        --block-background-fill: transparent !important;
        --block-background-fill-dark: transparent !important;
        --block-background-fill-light: transparent !important;
        --input-background-fill: rgba(255,255,255,0.06) !important;
        --input-background-fill-dark: rgba(255,255,255,0.06) !important;
        --border-color-primary: rgba(255,255,255,0.08) !important;
        --border-color-primary-dark: rgba(255,255,255,0.08) !important;
        --body-text-color: #e8edf5 !important;
        --body-text-color-dark: #e8edf5 !important;
        --block-label-text-color: rgba(255,255,255,0.55) !important;
        --block-label-text-color-dark: rgba(255,255,255,0.55) !important;
        --button-primary-background-fill: rgba(0, 162, 237, 0.7) !important;
        --button-primary-background-fill-dark: rgba(0, 162, 237, 0.7) !important;
        --button-secondary-background-fill: rgba(255,255,255,0.06) !important;
        --button-secondary-background-fill-dark: rgba(255,255,255,0.06) !important;
        --slider-color: #60a5fa !important;
        --slider-color-dark: #60a5fa !important;
        --checkbox-background-color-selected: #60a5fa !important;
        --checkbox-background-color-selected-dark: #60a5fa !important;
        --shadow-drop: 0 8px 32px rgba(0,0,0,0.25) !important;
        --shadow-drop-dark: 0 8px 32px rgba(0,0,0,0.25) !important;
    }

    /* Full-width container */
    html, body {
        max-width: 100vw !important;
        overflow-x: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    body.dark {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        background-attachment: fixed !important;
    }
    .gradio-container-outer {
        max-width: 100vw !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .gradio-container, .gradio-container .contain, .gradio-container .main-wrap {
        max-width: 100vw !important;
        width: 100vw !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        background: transparent !important;
    }
    .dark .gradio-container, .dark .gradio-container .contain, .dark .gradio-container .main-wrap {
        color: #e8edf5 !important;
    }
    
    /* Global font */
    body, input, button, select, textarea, span, p, div, label {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    /* Glass panels with frost effect (Dark Mode Only) */
    .dark .gr-group, .dark .group, .dark .form, .dark .block, .dark .panel, .dark .padded {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Navigation tabs (Dark Mode Only) */
    .dark .tabs {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 20px !important;
    }
    
    .tab-nav {
        border-bottom: none !important;
        display: flex !important;
        gap: 4px !important;
    }
    
    .dark .tab-nav button {
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: rgba(255, 255, 255, 0.5) !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.03em !important;
    }
    
    .dark .tab-nav button.selected {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Glass buttons */
    .primary-btn {
        background: rgba(0, 162, 237, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(0, 162, 237, 0.25) !important;
    }
    .primary-btn:hover {
        background: rgba(0, 162, 237, 0.85) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 162, 237, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    .primary-btn:active {
        transform: translateY(0px) !important;
    }
    
    .dark .secondary-btn {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    .dark .secondary-btn:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    .dark .secondary-btn:active {
        transform: translateY(0px) !important;
    }
    
    /* Glass accordions (Dark Mode Only) */
    .dark .accordion {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        overflow: visible !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        transition: border-color 0.2s ease !important;
    }
    .dark .accordion:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    
    .dark .accordion > summary, .dark .accordion .label-wrap {
        background: rgba(255, 255, 255, 0.04) !important;
        padding: 8px 14px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px 10px 0 0 !important;
    }
    
    /* Develop panel container */
    .develop-panel {
        max-height: 84vh !important;
        overflow-y: auto !important;
        padding-right: 6px !important;
        background: transparent !important;
        border: none !important;
    }
    
    .develop-panel::-webkit-scrollbar { width: 4px !important; }
    .develop-panel::-webkit-scrollbar-track { background: transparent !important; }
    .develop-panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15) !important; border-radius: 2px !important; }
    .develop-panel::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25) !important; }
    .develop-panel { scrollbar-width: thin !important; scrollbar-color: rgba(255,255,255,0.15) transparent !important; }
    
    /* Sliders */
    .dark .gr-slider input[type=range] {
        accent-color: #60a5fa !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Text inputs (Dark Mode Only) */
    .dark input[type="text"], .dark input[type="number"], .dark select, .dark textarea {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        color: #e8edf5 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    .dark input[type="text"]:focus, .dark input[type="number"]:focus, .dark select:focus, .dark textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Checkbox & Radio */
    input[type="checkbox"] { accent-color: #60a5fa !important; }
    
    /* Typography (Dark Mode Only) */
    .dark h1, .dark h2, .dark h3,
    .dark h4, .dark h5, .dark h6,
    .dark p, .dark strong, .dark .prose,
    .dark .prose h1, .dark .prose h2,
    .dark .prose h3, .dark .prose h4,
    .dark .prose p, .dark .markdown-text h1,
    .dark .markdown-text h2, .dark .markdown-text h3,
    .dark .markdown-text p, .dark div.markdown {
        color: #e8edf5 !important;
    }
    
    /* Form labels (Dark Mode Only) */
    .dark label span,
    .dark .form-label,
    .dark label .form-label-text,
    .dark .label-val {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        color: rgba(255, 255, 255, 0.55) !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
    }
    
    /* Preset chips (glass style - Dark Mode Only) */
    .preset-chips { border: none !important; background: transparent !important; padding: 0 !important; }
    .preset-chips .wrap {
        display: flex !important;
        flex-direction: column !important;
        max-height: 380px !important;
        overflow-y: auto !important;
        gap: 4px !important;
        background: transparent !important;
        border: none !important;
        padding: 0 4px 0 0 !important;
    }
    .preset-chips .wrap::-webkit-scrollbar { width: 3px !important; }
    .preset-chips .wrap::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12) !important; border-radius: 2px !important; }
    
    .dark .preset-chips label {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        color: rgba(255, 255, 255, 0.6) !important;
        width: 100% !important;
    }
    .dark .preset-chips label:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: rgba(255, 255, 255, 0.85) !important;
        transform: translateY(-1px) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    .dark .preset-chips label.selected {
        background: rgba(96, 165, 250, 0.15) !important;
        color: #93c5fd !important;
        border-left: 3px solid #60a5fa !important;
        border-radius: 0 8px 8px 0 !important;
    }
    .preset-chips input[type="radio"] { display: none !important; }
    .preset-chips label .radio-circle { display: none !important; }
    
    /* Dropdown options (Dark Mode Only) */
    .dark ul.options, .dark .options {
        background: rgba(30, 27, 75, 0.95) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        z-index: 9999 !important;
        position: absolute !important;
        overflow-y: auto !important;
        max-height: 280px !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(255,255,255,0.15) rgba(255,255,255,0.02) !important;
    }
    .dark ul.options > li, .dark .options > li {
        color: rgba(255, 255, 255, 0.7) !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    .dark ul.options > li:hover, .dark .options > li:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #ffffff !important;
    }
    .dark ul.options > li.selected, .dark .options > li.selected {
        color: #93c5fd !important;
        background: rgba(96, 165, 250, 0.12) !important;
    }
    
    /* Click-to-zoom */
    #retouch-output { cursor: zoom-in !important; }
    .cmp-label { font-weight: 600 !important; }
    #retouch-compare { cursor: ew-resize !important; }
    
    /* File upload */
    .dark .gr-file {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    .dark .gr-file:hover { border-color: rgba(96, 165, 250, 0.5) !important; }
    
    /* Overflow fix */
    .gradio-container .block, .gradio-container .wrap,
    .gradio-container .group, .gradio-container .form,
    .gradio-container .row, .gradio-container .col,
    .gradio-container .column, .gradio-container .panel,
    .gradio-container .padded, .gradio-container .tabs,
    .gradio-container .tabitem, .gradio-container .dropdown,
    .gradio-container .dropdown-container {
        overflow: visible !important;
    }

    /* Re-enable internal scroll on develop-panel after the overflow fix above.
       The .gradio-container .column rule (specificity 0,2,0) overrides
       .develop-panel (0,1,0), killing overflow-y:auto and clipping content
       past max-height:84vh. This higher-specificity rule restores it. */
    .gradio-container .column.develop-panel {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* Header Bar styling adapting to both Light and Dark mode */
    .header-bar {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        padding: 0.6rem 1.5rem !important;
        background: rgba(0, 0, 0, 0.03) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 18px !important;
        font-family: -apple-system, sans-serif !important;
        border-radius: 14px !important;
    }
    .header-left {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    .header-badge {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: #ffffff !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px !important;
    }
    .header-title {
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: rgba(15, 23, 42, 0.9) !important;
        letter-spacing: 0.3px !important;
    }
    .header-version {
        font-size: 0.7rem !important;
        color: rgba(15, 23, 42, 0.4) !important;
        border-left: 1px solid rgba(15, 23, 42, 0.1) !important;
        padding-left: 10px !important;
        margin-left: 2px !important;
        font-weight: 500 !important;
    }
    .header-workspace {
        font-size: 0.75rem !important;
        color: rgba(15, 23, 42, 0.5) !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
    }

    /* Dark mode overrides for Header Bar */
    .dark .header-bar {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .dark .header-badge {
        background: linear-gradient(135deg, #60a5fa, #a78bfa) !important;
        color: #ffffff !important;
    }
    .dark .header-title {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    .dark .header-version {
        color: rgba(255, 255, 255, 0.3) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .dark .header-workspace {
        color: rgba(255, 255, 255, 0.35) !important;
    }
""", head="""
    <script>
    (function() {
        // Keyboard Shortcuts
        document.addEventListener('keydown', function(e) {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                var btn = document.querySelector('.primary-btn');
                if (btn) btn.click();
            }
            if ((e.metaKey || e.ctrlKey) && e.key === 'r') {
                e.preventDefault();
                var openDetails = document.querySelector('.develop-panel details[open]');
                if (openDetails) {
                    var resetBtn = openDetails.querySelector('.section-reset-btn');
                    if (resetBtn) resetBtn.click();
                }
            }
        });

        // Click-to-Zoom Modal
        function openZoomModal(src) {
            var overlay = document.createElement('div');
            overlay.id = 'zoom-overlay';
            overlay.style.position = 'fixed';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100vw';
            overlay.style.height = '100vh';
            overlay.style.backgroundColor = 'rgba(10, 8, 25, 0.95)';
            overlay.style.backdropFilter = 'blur(24px)';
            overlay.style.webkitBackdropFilter = 'blur(24px)';
            overlay.style.zIndex = '99999';
            overlay.style.display = 'flex';
            overlay.style.alignItems = 'center';
            overlay.style.justifyContent = 'center';
            overlay.style.cursor = 'zoom-out';
            overlay.style.opacity = '0';
            overlay.style.transition = 'opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1)';
            
            var img = document.createElement('img');
            img.src = src;
            img.style.maxHeight = '92vh';
            img.style.maxWidth = '92vw';
            img.style.objectFit = 'contain';
            img.style.borderRadius = '8px';
            img.style.boxShadow = '0 24px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)';
            img.style.transform = 'scale(0.95)';
            img.style.transition = 'transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)';
            
            overlay.appendChild(img);
            document.body.appendChild(overlay);
            
            setTimeout(function() {
                overlay.style.opacity = '1';
                img.style.transform = 'scale(1)';
            }, 10);
            
            overlay.addEventListener('click', function() {
                overlay.style.opacity = '0';
                img.style.transform = 'scale(0.95)';
                setTimeout(function() {
                    overlay.remove();
                }, 250);
            });
        }

        document.addEventListener('click', function(e) {
            var target = e.target;
            if (target.tagName === 'IMG' && (target.closest('#retouch-output') || target.closest('#retouch-compare'))) {
                e.preventDefault();
                openZoomModal(target.src);
            }
        });

        // Theme management and layout helpers
        function forceFullWidth() {
            document.querySelectorAll('.gradio-container-outer, .gradio-container').forEach(function(el){
                if (el.style.maxWidth !== 'none') {
                    el.style.setProperty('max-width', 'none', 'important');
                }
                if (el.style.width !== '100vw') {
                    el.style.setProperty('width', '100vw', 'important');
                }
                if (el.style.minWidth !== '100vw') {
                    el.style.setProperty('min-width', '100vw', 'important');
                }
            });
        }
        function forceDarkMode() {
            var isDark = window.location.search.includes('__theme=dark') || localStorage.getItem('theme') === 'dark' || window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (isDark) {
                if (!document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.add('dark');
                }
                if (document.body && !document.body.classList.contains('dark')) {
                    document.body.classList.add('dark');
                }
                document.querySelectorAll('.gradio-container, .gradio-container-outer').forEach(function(el){
                    if (!el.classList.contains('dark')) {
                        el.classList.add('dark');
                    }
                });
            } else {
                if (document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.remove('dark');
                }
                if (document.body && document.body.classList.contains('dark')) {
                    document.body.classList.remove('dark');
                }
                document.querySelectorAll('.gradio-container, .gradio-container-outer').forEach(function(el){
                    if (el.classList.contains('dark')) {
                        el.classList.remove('dark');
                    }
                });
            }
        }
        
        var observer = new MutationObserver(function(){
            observer.disconnect();
            forceFullWidth();
            forceDarkMode();
            observer.observe(document.documentElement, {attributes: true, subtree: true, attributeFilter: ['style', 'class']});
        });
        observer.observe(document.documentElement, {attributes: true, subtree: true, attributeFilter: ['style', 'class']});
        ['load', 'DOMContentLoaded', 'gradio:ready'].forEach(function(e){ window.addEventListener(e, function(){ forceFullWidth(); forceDarkMode(); }); });
        setTimeout(function(){ forceFullWidth(); forceDarkMode(); }, 200);
        setTimeout(function(){ forceFullWidth(); forceDarkMode(); }, 1000);
        setTimeout(function(){ forceFullWidth(); forceDarkMode(); }, 3000);

        // Comparison Slider: drag handle + syncSize (delegated).
        // NOTE: COMPARE_TPL's <script> cannot run because Gradio's gr.HTML
        // injects the value via element.innerHTML, and HTML5 spec says
        // scripts inserted that way are inert. So the drag/sync logic lives
        // here and uses event delegation on document so listeners survive
        // every innerHTML replacement.
        (function() {
            function findCmp(node) {
                if (!node) return null;
                if (node.id && node.id.indexOf('cmp-') === 0) return node;
                var c = node.closest && node.closest('[id^="cmp-"]');
                if (c) return c;
                var byId = node.querySelector && node.querySelector('[id^="cmp-"]');
                return byId || null;
            }

            function setupCmp(c) {
                if (!c || c.__cmpInited) return;
                c.__cmpInited = true;
                var o = c.querySelector('.cmp-overlay');
                var imgResult = c.children[0];
                var imgOrig = o && o.children[0];
                if (!o || !imgResult || !imgOrig) return;

                function syncSize() {
                    var rect = imgResult.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        imgOrig.style.width = rect.width + 'px';
                        imgOrig.style.height = rect.height + 'px';
                    }
                }
                if (imgResult.complete && imgResult.naturalWidth > 0) {
                    syncSize();
                } else {
                    imgResult.addEventListener('load', syncSize);
                }
                window.addEventListener('resize', syncSize);
                setTimeout(syncSize, 50);
                setTimeout(syncSize, 200);
                setTimeout(syncSize, 1000);
            }

            var activeDrag = null;

            function beginDrag(handle, e) {
                var c = findCmp(handle);
                if (!c) return;
                setupCmp(c);
                var o = c.querySelector('.cmp-overlay');
                var h = c.querySelector('.cmp-handle');
                var imgResult = c.children[0];
                var imgOrig = o.children[0];

                function syncSize() {
                    var rect = imgResult.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        imgOrig.style.width = rect.width + 'px';
                        imgOrig.style.height = rect.height + 'px';
                    }
                }
                function move(x) {
                    var r = c.getBoundingClientRect();
                    var p = Math.max(0, Math.min(100, (x - r.left) / r.width * 100));
                    o.style.width = p + '%';
                    h.style.left = p + '%';
                    syncSize();
                }
                activeDrag = { move: move };
                if (e && e.preventDefault) e.preventDefault();
            }

            document.addEventListener('mousedown', function(e) {
                var handle = e.target.closest && e.target.closest('.cmp-handle');
                if (handle) beginDrag(handle, e);
            });
            document.addEventListener('mousemove', function(e) {
                if (activeDrag) activeDrag.move(e.clientX);
            });
            document.addEventListener('mouseup', function() { activeDrag = null; });

            document.addEventListener('touchstart', function(e) {
                var handle = e.target.closest && e.target.closest('.cmp-handle');
                if (handle) beginDrag(handle, e);
            }, { passive: false });
            document.addEventListener('touchmove', function(e) {
                if (activeDrag && e.touches[0]) activeDrag.move(e.touches[0].clientX);
            }, { passive: false });
            document.addEventListener('touchend', function() { activeDrag = null; });

            // Auto-setup whenever a new comparison is inserted.
            var mo = new MutationObserver(function(muts) {
                for (var i = 0; i < muts.length; i++) {
                    var added = muts[i].addedNodes;
                    for (var j = 0; j < added.length; j++) {
                        var n = added[j];
                        if (n.nodeType !== 1) continue;
                        var c = findCmp(n);
                        if (c) setupCmp(c);
                    }
                }
            });
            mo.observe(document.body, { childList: true, subtree: true });
        })();
    })();
    </script>
""") as app:

    # Liquid Glass Header bar
    gr.HTML("""
    <div class="header-bar">
        <div class="header-left">
            <span class="header-badge">RP</span>
            <span class="header-title">Retouch Pro</span>
            <span class="header-version">v2.0.0</span>
        </div>
        <div class="header-workspace">
            Liquid Glass Workspace
        </div>
    </div>
    """)


    with gr.Tabs():
        with gr.Tab("Single Photo Editor"):
            with gr.Row():
                # Column 1: Presets & Library Panel (Left)
                with gr.Column(scale=2, elem_classes=["library-panel"]):
                    img_input = gr.File(label="Input Image(s) (RAW supported)", file_types=["image"], file_count="multiple")

                    with gr.Group():
                        recipe = gr.Radio(
                            choices=RECIPE_NAMES, value="natural", label="Base Preset Recipe",
                            info="Select a preset recipe to auto-fill sliders",
                            elem_classes=["preset-chips"]
                        )
                        custom_style_preset = gr.Dropdown(
                            choices=custom_style_choices, value=None, label="Or Load Custom Style Profile", interactive=True,
                            info="Select an extracted style from your library"
                        )

                        show_compare = gr.Checkbox(label="Show side-by-side comparison screen", value=True, info="Split view: original | separator | retouched result")

                        with gr.Row():
                            fast = gr.Checkbox(label="Fast Preview (Recommended)", value=True,
                                               info="Process at half resolution for preview; final export always uses full quality")
                        
                        with gr.Row():
                            process_btn = gr.Button("Process Image(s) ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])
                            reset_btn = gr.Button("Reload Recipe Defaults 🔄", variant="secondary", size="lg", elem_classes=["secondary-btn"], elem_id="reset-btn")

                    with gr.Group():
                        gr.Markdown("### ⚙️ Export Settings")
                        with gr.Row():
                            export_fmt = gr.Radio(choices=["JPEG", "PNG", "WebP"], value="JPEG", label="Format", interactive=True)
                            export_quality = gr.Slider(10, 100, 95, step=1, label="Compression Quality", info="For JPEG/WebP formats")
                        export_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Resize / Limit Resolution", interactive=True,
                            info="Downscales image if it exceeds target dimension while maintaining aspect ratio"
                        )

                # Column 2: Workspace Canvas (Center)
                with gr.Column(scale=4, elem_classes=["viewer-panel"]):
                    with gr.Group():
                        gr.Markdown("### 🖼️ Preview Canvas")
                        img_output = gr.Image(height=600, show_label=False, elem_id="retouch-output")
                        compare_viewer = gr.HTML(visible=False, elem_id="retouch-compare")
                        _original_state = gr.State(value=None)
                        status = gr.Textbox(label="Status", interactive=False, placeholder="Upload an image and click Process to start...")
                        export_file = gr.File(label="📥 Download Exported Assets")

                    with gr.Group(visible=False) as debug_panel:
                        gr.Markdown("### 🔍 Debug Masks & Frequency Layers")
                        debug_gallery = gr.Gallery(label="Masks (skin, skin+hair, lips, sharpen, glow, freq_low, freq_mid, freq_high)", columns=4, height=300)

                # Column 3: Adjustment Panel (Right)
                with gr.Column(scale=3, elem_classes=["develop-panel"]):
                    with gr.Group():
                        gr.Markdown("### ⚙️ Develop Adjustments")
                        
                        with gr.Accordion("✨ Skin Smoothing & Texture", open=True):
                            reset_skin_smooth_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            smooth = gr.Slider(0, 100, 30, step=1, label="Smooth", info="Strength of skin smoothing (blur/median blend)")
                            nose_smooth = gr.Slider(0, 100, 0, step=1, label="Nose Smooth (0 = follow face)", info="Additional smoothing for nose bridge highlights")
                            mid_reduction = gr.Slider(0.0, 1.0, 0.45, step=0.05, label="Mid Frequency Reduction", info="Target mid-level skin blemishes while preserving high-frequency pores")
                            texture_opacity = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Texture Opacity", info="Control original pore structure opacity overlay")
                            micro_restore = gr.Slider(0, 50, 20, step=1, label="Micro-Texture Restore", info="Re-inject dimensional micro-contrast in cheek/nose/under-eye zones after smoothing (0 = off, 25 = subtle, 50 = strong)")
                            pore_synthesis = gr.Slider(0, 100, 0, step=1, label="Pore Synthesis", info="Add micro-texture/synthesized pores to prevent artificial plastic skin")
                            blemish = gr.Slider(0, 100, 30, step=1, label="Blemish Removal", info="AI blemish detection and inpainting for acne/spots")

                        with gr.Accordion("🎨 Skin Tone", open=False):
                            reset_skin_tone_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            whiten = gr.Slider(0, 100, 10, step=1, label="Whitening", info="Luminance boost and porcelain skin color match")
                            whiten_tone = gr.Dropdown(choices=WHITEN_TONE_CHOICES, value="rosy", label="Whitening Tone", interactive=True, info="Tone direction: rosy (warm pink), porcelain (cool neutral), neutral")
                            equalize = gr.Slider(0, 100, 20, step=1, label="Equalize", info="Even out skin redness and regional color inconsistencies")
                            auto_exposure = gr.Checkbox(label="Auto Exposure Correction", value=False, info="Automatically correct under/over-exposed images before processing")
                            white_costume_lift = gr.Checkbox(label="White Costume Lift", value=False, info="Selectively boost bright clothing to create separation")

                        with gr.Accordion("📊 Basic Tone & Color", open=False):
                            reset_basic_tone_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            contrast = gr.Slider(-50, 50, 0, step=1, label="Contrast", info="Adjust global image contrast")
                            brightness = gr.Slider(-50, 50, 0, step=1, label="Brightness", info="Adjust global image brightness")
                            clarity = gr.Slider(-100, 100, 0, step=1, label="Clarity", info="Mid-tone contrast / local contrast enhancement (negative = soften)")
                            vibrance = gr.Slider(-100, 100, 0, step=1, label="Vibrance", info="Smart saturation boost that protects skin tones")
                            saturation = gr.Slider(-100, 100, 0, step=1, label="Saturation", info="Uniform global saturation adjustment")

                        with gr.Accordion("📈 Tone Curve", open=False):
                            reset_tone_curve_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            highlights = gr.Slider(-100, 100, 0, step=1, label="Highlights", info="Recover or boost bright highlight regions")
                            shadows = gr.Slider(-100, 100, 0, step=1, label="Shadows", info="Open up or deepen shadow regions")
                            whites = gr.Slider(-100, 100, 0, step=1, label="Whites", info="Control absolute white point ceiling")
                            blacks = gr.Slider(-100, 100, 0, step=1, label="Blacks", info="Control absolute black point floor")

                        with gr.Accordion("💡 Virtual Studio Relighting", open=False):
                            reset_relighting_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            relight = gr.Slider(0, 100, 0, step=1, label="Relight Strength", info="Intensity of 3D virtual studio light source redirection")
                            relight_azimuth = gr.Slider(-180, 180, 0, step=1, label="Light Azimuth", info="Horizontal light source direction angle (-180° to 180°)")
                            relight_elevation = gr.Slider(-90, 90, 30, step=1, label="Light Elevation", info="Vertical light source direction angle (-90° to 90°)")

                        with gr.Accordion("👁️ Eyes & Lips", open=False):
                            reset_eyes_lips_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            eye_enhance = gr.Slider(0, 100, 5, step=1, label="Eye Enhance", info="Boost eye clarity, iris reflection details, and whites brightness")
                            catchlight = gr.Slider(0, 100, 0, step=1, label="Catchlight Boost", info="Amplify existing catchlight highlights in the iris (0 = follow Eye Enhance)")
                            dark_circles = gr.Slider(0, 100, 0, step=1, label="Dark Circle Repair", info="Under-eye dark circle detection and repair")
                            teeth_whiten = gr.Slider(0, 100, 5, step=1, label="Teeth Whiten", info="Naturally whiten and brighten teeth enamel")
                            lip_enhance = gr.Slider(0, 100, 5, step=1, label="Lip Enhance", info="Enhance lip texture definition, gloss, and contour")
                            lip_tint = gr.Dropdown(choices=LIP_TINTS, value="none", label="Lip Tint Color", interactive=True, info="Apply a natural cosmetic tint overlay")
                            lip_finish = gr.Dropdown(choices=LIP_FINISH_CHOICES, value="gloss", label="Lip Finish", interactive=True, info="Surface finish style: gloss (shiny), matte (flat), velvet (soft)")
                            blush = gr.Slider(0, 100, 0, step=1, label="Blush Strength", info="Intensity of virtual cosmetic blush on cheeks")
                            with gr.Row():
                                nose_blush = gr.Checkbox(label="Nose Blush", value=False, info="Add cosmetic pink tone to nose tip")
                                under_eye_blush = gr.Checkbox(label="Under-Eye Blush", value=False, info="Apply soft under-eye blush for a fresh/cosplay look")

                        with gr.Accordion("🧬 Face Reshaping", open=False):
                            reset_face_reshaping_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            slimming = gr.Slider(0, 100, 0, step=1, label="Face Slimming", info="Liquify-based face slimming/reshaping via landmark-driven warp")

                        with gr.Accordion("🌟 Structure & Effects", open=False):
                            reset_structure_effects_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            hair_enhance = gr.Slider(0, 100, 5, step=1, label="Hair Shine", info="Boost highlight reflections and depth in hair strands")
                            dodge_burn = gr.Slider(0, 100, 0, step=1, label="Dodge & Burn", info="Sculpt face structure with local highlight/shadow contouring")
                            impact = gr.Slider(0, 100, 0, step=1, label="Global Impact Finish", info="Final punch: combined clarity, sharpening, and micro-contrast boost")
                            specular_bloom = gr.Slider(0, 100, 0, step=1, label="Specular Bloom", info="Dreamy bloom glow applied specifically to skin highlight zones")
                            specular_bloom_tone = gr.Dropdown(choices=SPECULAR_BLOOM_TONE_CHOICES, value="rosy", label="Specular Bloom Tone", interactive=True, info="Color tint of the specular bloom glow")
                            bloom = gr.Slider(0, 100, 0, step=1, label="Orton Bloom (Overall Glow)", info="High-key glow blending for high-fashion portraits")
                            bloom_threshold = gr.Slider(150, 250, 210, step=1, label="Bloom Threshold", info="Brightness threshold where the glow begins to bleed")
                            bloom_softness = gr.Slider(1, 100, 30, step=1, label="Bloom Softness", info="Softness blur radius of the bloom filter")
                            sharpen = gr.Slider(0, 100, 0, step=1, label="Selective Sharpening", info="Sharpen eyes, eyebrows, and hair edges (mask-driven)")
                            sharpen_radius = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="Sharpen Radius", info="Blur radius for unsharp mask kernel")
                            glow = gr.Slider(0, 100, 0, step=1, label="Atmospheric Glow", info="Multi-scale atmospheric glow/bloom effect")
                            vignette = gr.Slider(0, 100, 0, step=1, label="Vignette", info="Darken image corners for a focused portrait look")
                            subject_separation = gr.Slider(0, 100, 0, step=1, label="Subject-Background Separation", info="Brighten subject / darken background using person segmentation mask")

                        with gr.Accordion("🎬 Film Color Grading", open=False):
                            reset_color_grading_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            color_grade = gr.Dropdown(choices=COLOR_GRADE_NAMES, value="natural", label="Color Grade Preset", interactive=True, info="Apply a film/color grading preset from the presets library")
                            grade_intensity = gr.Slider(0, 100, 0, step=1, label="Grade Intensity", info="Blend strength of the color grade (0-100%)")

                        with gr.Accordion("🎞️ Film & Analog Effects", open=False):
                            reset_film_effects_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            chromatic_aberration = gr.Slider(0, 20, 0, step=0.5, label="Chromatic Aberration", info="Lens fringing effect (RGB channel shift in pixels)")
                            grain = gr.Slider(0, 100, 0, step=1, label="Film Grain", info="Analog film grain noise overlay (0-100 maps to engine 0.0-0.2)")
                            halation = gr.Slider(0, 100, 0, step=1, label="Halation", info="Red light bloom around bright highlights (0-100 maps to engine 0.0-1.0)")
                            tonal_curve_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Tonal Curve", info="Film H&D tonal curve strength (lifted blacks + S-curve)")
                            skin_protect_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Skin Protection", info="Preserve skin hues during color grading ops")
                            grain_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Organic Grain", info="Clumped luminance-correlated film grain (Fuji-style)")
                            highlight_rolloff_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Highlight Rolloff", info="Soft C¹-continuous highlight compression")
                            lut = gr.Dropdown(choices=LUT_CHOICES, value="none", label="Film Emulation LUT", interactive=True, info="Apply a film stock emulation LUT (Kodak / Fuji)")

                        with gr.Accordion("🌈 Split Toning", open=False):
                            reset_split_toning_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("**Shadows**")
                            shadow_hue = gr.Slider(0, 360, 0, step=1, label="Shadow Hue", info="Hue shift applied to shadow tones (degrees)")
                            shadow_sat = gr.Slider(0, 100, 0, step=1, label="Shadow Saturation", info="Saturation boost for shadow tones")
                            gr.Markdown("**Midtones**")
                            midtone_hue = gr.Slider(0, 360, 0, step=1, label="Midtone Hue", info="Hue shift applied to midtone tones (degrees)")
                            midtone_sat = gr.Slider(0, 100, 0, step=1, label="Midtone Saturation", info="Saturation boost for midtone tones")
                            gr.Markdown("**Highlights**")
                            highlight_hue = gr.Slider(0, 360, 0, step=1, label="Highlight Hue", info="Hue shift applied to highlight tones (degrees)")
                            highlight_sat = gr.Slider(0, 100, 0, step=1, label="Highlight Saturation", info="Saturation boost for highlight tones")

                        with gr.Accordion("🔮 Color Transfer", open=False):
                            reset_color_transfer_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("Upload a reference image to match its color tone using CDF-based histogram transfer")
                            color_ref_img = gr.Image(type="filepath", label="Reference Image", show_label=True, height=160)
                            color_ref_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Transfer Strength", info="Mix ratio between original grade and matched reference grade (1.0 = full transfer, 0.0 = no transfer)")

                        with gr.Accordion("🔍 Debug & Mask Preview", open=False):
                            reset_debug_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            debug_mode = gr.Checkbox(label="Generate Debug Masks", value=False, info="Save skin/lips/frequency-layer masks and display them for tuning")

                        process_btn_bottom = gr.Button("Apply Overrides & Process ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])

                        gr.HTML("""
                        <div style="margin-top:12px;text-align:center;font-size:0.7rem;color:rgba(255,255,255,0.4);border-top:1px solid rgba(255,255,255,0.06);padding-top:10px">
                            <span>⌘+Enter Process · ⌘+R Reset · Click preview for full-size</span>
                        </div>
                        """)


        with gr.Tab("Batch Library Ingestion"):
            with gr.Row():
                with gr.Column(scale=1):
                    folder_in = gr.Textbox(label="Input Folder Path", placeholder="/path/to/photos", info="Absolute path to directory containing raw/jpeg source photos.")
                    folder_out = gr.Textbox(label="Output Folder Path", placeholder="/path/to/exports", info="Absolute path where processed results will be written.")
                    
                    with gr.Group():
                        gr.Markdown("### Style Mode")
                        batch_style_type = gr.Radio(choices=["Use Standard Recipe", "Use Custom Style"], value="Use Standard Recipe", label="Style Mode", info="Choose whether to apply a built-in recipe preset or a custom learned style profile.")
                        batch_recipe = gr.Dropdown(choices=RECIPE_NAMES, value="natural", label="Standard Recipe", info="Select standard built-in recipe preset.")
                        batch_custom_style = gr.Dropdown(choices=custom_style_choices, value=None, label="Custom Style Profile", interactive=True, visible=False, info="Select a custom style profile from your library.")
                    
                    with gr.Group():
                        gr.Markdown("### Export Formatting")
                        batch_fmt = gr.Radio(choices=["JPEG", "PNG", "WebP"], value="JPEG", label="Format", interactive=True)
                        batch_quality = gr.Slider(10, 100, 95, step=1, label="Quality")
                        batch_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Export Resolution", interactive=True
                        )
                        
                    with gr.Row():
                        auto_group_toggle = gr.Checkbox(label="Enable Rule-Based Auto-Grouping", value=True, info="Group similar scenes to ensure visual coherence across outputs.")
                        sheet_toggle = gr.Checkbox(label="Generate Contact Sheet", value=True, info="Generate a printable contact grid sheet for all processed photos.")
                        zip_toggle = gr.Checkbox(label="Package into ZIP", value=True, info="Archive all output files into a single downloadable .zip file.")
                        
                    batch_btn = gr.Button("Process Entire Folder 🚀", variant="primary", size="lg", elem_classes=["primary-btn"])
                    
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 📋 Automation Output")
                        batch_sheet_out = gr.Image(label="Generated Contact Sheet", height=320)
                        batch_zip_out = gr.File(label="Download Packaged ZIP")
                        batch_status = gr.Textbox(label="Execution Log & Statistics", lines=12, interactive=False, placeholder="Click 'Process Entire Folder' to start batch processing...")

        with gr.Tab("Custom Style Library"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Save Sliders as Custom Style")
                    save_name = gr.Textbox(label="Style Name", placeholder="e.g. Dennis Cosplay v4", info="Give your custom style preset a unique name.")
                    save_author = gr.Textbox(label="Author", value="Dennis")
                    save_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="moody, cosplay, soft")
                    save_style_btn = gr.Button("Save Sliders to Style Library 💾", variant="primary", elem_classes=["primary-btn"])
                    save_status = gr.Textbox(label="Save Status", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Style Dataset Learning (Pairs Extractor)")
                    learn_orig_dir = gr.Textbox(label="Original Folder Path", placeholder="/path/to/originals", info="Directory containing original, un-retouched photos.")
                    learn_edit_dir = gr.Textbox(label="Edited Folder Path", placeholder="/path/to/edited", info="Directory containing matching edited/retouched photos (same filenames).")
                    learn_name = gr.Textbox(label="Learned Style Name", placeholder="e.g. Dennis_Cosplay_V7", info="Unique name for the learned style profile.")
                    learn_author = gr.Textbox(label="Author", value="Dennis")
                    learn_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="learned, cosplay")
                    learn_style_btn = gr.Button("Extract & Learn Style from Dataset 🧠", variant="primary", elem_classes=["primary-btn"])
                    learn_status = gr.Textbox(label="Learning Status", lines=5, interactive=False)

    def on_batch_style_change(style_type):
        if style_type == "Use Custom Style":
            return [gr.update(visible=False), gr.update(visible=True)]
        else:
            return [gr.update(visible=True), gr.update(visible=False)]

    # Event binding setup
    _recipe_outputs = [
        smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth, micro_restore,
        whiten, equalize, blemish, whiten_tone, nose_blush, under_eye_blush, white_costume_lift,
        dodge_burn, relight, relight_azimuth, relight_elevation, specular_bloom, specular_bloom_tone,
        eye_enhance, catchlight, dark_circles, teeth_whiten, lip_enhance, lip_tint, lip_finish, blush, slimming, hair_enhance,
        contrast, brightness, highlights, shadows, whites, blacks, clarity, vibrance, saturation, auto_exposure,
        bloom, bloom_threshold, bloom_softness, glow, vignette, sharpen, sharpen_radius, subject_separation, impact,
        color_grade, grade_intensity,
        chromatic_aberration, grain, halation, lut,
        tonal_curve_strength, skin_protect_strength, grain_strength, highlight_rolloff_strength,
        shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat,
    ]

    recipe.change(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=_recipe_outputs,
    )

    custom_style_preset.change(
        fn=apply_custom_style,
        inputs=[custom_style_preset, recipe],
        outputs=_recipe_outputs,
    )

    reset_btn.click(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=_recipe_outputs,
    )

    reset_skin_smooth_btn.click(
        fn=reset_skin_smoothing,
        inputs=[recipe],
        outputs=[smooth, nose_smooth, mid_reduction, texture_opacity, micro_restore, pore_synthesis, blemish]
    )

    reset_skin_tone_btn.click(
        fn=reset_skin_tone,
        inputs=[recipe],
        outputs=[whiten, whiten_tone, equalize, auto_exposure, white_costume_lift]
    )

    reset_basic_tone_btn.click(
        fn=reset_basic_tone,
        inputs=[recipe],
        outputs=[contrast, brightness, clarity, vibrance, saturation]
    )

    reset_tone_curve_btn.click(
        fn=reset_tone_curve,
        inputs=[recipe],
        outputs=[highlights, shadows, whites, blacks]
    )

    reset_relighting_btn.click(
        fn=reset_relighting,
        inputs=[recipe],
        outputs=[relight, relight_azimuth, relight_elevation]
    )

    reset_eyes_lips_btn.click(
        fn=reset_eyes_lips,
        inputs=[recipe],
        outputs=[eye_enhance, catchlight, dark_circles, teeth_whiten, lip_enhance, lip_tint, lip_finish, blush, nose_blush, under_eye_blush]
    )

    reset_face_reshaping_btn.click(
        fn=reset_face_reshaping,
        inputs=[recipe],
        outputs=[slimming]
    )

    reset_structure_effects_btn.click(
        fn=reset_structure_effects,
        inputs=[recipe],
        outputs=[hair_enhance, dodge_burn, impact, specular_bloom, specular_bloom_tone, bloom, bloom_threshold, bloom_softness, sharpen, sharpen_radius, glow, vignette, subject_separation]
    )

    reset_color_grading_btn.click(
        fn=reset_color_grading,
        inputs=[recipe],
        outputs=[color_grade, grade_intensity]
    )

    reset_film_effects_btn.click(
        fn=reset_film_effects,
        inputs=[recipe],
        outputs=[chromatic_aberration, grain, halation, lut, tonal_curve_strength, skin_protect_strength, grain_strength, highlight_rolloff_strength]
    )

    reset_split_toning_btn.click(
        fn=reset_split_toning,
        inputs=[recipe],
        outputs=[shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat]
    )

    reset_color_transfer_btn.click(
        fn=reset_color_transfer,
        inputs=[],
        outputs=[color_ref_img, color_ref_strength]
    )

    reset_debug_btn.click(
        fn=reset_debug,
        inputs=[recipe],
        outputs=[debug_mode]
    )


    save_style_btn.click(
        fn=on_save_style,
        inputs=[save_name, save_author, save_tags,
                smooth, mid_reduction, texture_opacity,
                whiten, contrast, brightness],
        outputs=[custom_style_preset, batch_custom_style, save_status]
    )

    learn_style_btn.click(
        fn=on_learn_style,
        inputs=[learn_orig_dir, learn_edit_dir, learn_name, learn_author, learn_tags],
        outputs=[custom_style_preset, batch_custom_style, learn_status]
    )

    batch_style_type.change(
        fn=on_batch_style_change,
        inputs=[batch_style_type],
        outputs=[batch_recipe, batch_custom_style],
    )

    batch_btn.click(
        fn=on_process_folder,
        inputs=[folder_in, folder_out, batch_style_type, batch_custom_style, batch_recipe,
                batch_fmt, batch_quality, batch_res, auto_group_toggle, sheet_toggle, zip_toggle],
        outputs=[batch_sheet_out, batch_zip_out, batch_status]
    )

    _process_inputs = [
        img_input, recipe,
        smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth, micro_restore,
        whiten, equalize, blemish, whiten_tone, nose_blush, under_eye_blush, white_costume_lift,
        dodge_burn, relight, relight_azimuth, relight_elevation, specular_bloom, specular_bloom_tone,
        eye_enhance, catchlight, dark_circles, teeth_whiten, lip_enhance, lip_tint, lip_finish, blush, slimming, hair_enhance,
        contrast, brightness, highlights, shadows, whites, blacks, clarity, vibrance, saturation, auto_exposure,
        bloom, bloom_threshold, bloom_softness, glow, vignette, sharpen, sharpen_radius, subject_separation, impact,
        color_grade, grade_intensity, chromatic_aberration, grain, halation, lut,
        tonal_curve_strength, skin_protect_strength, grain_strength, highlight_rolloff_strength,
        shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat,
        color_ref_img, color_ref_strength,
        show_compare, fast,
        export_fmt, export_quality, export_res,
        debug_mode,
    ]
    _process_outputs = [img_output, compare_viewer, _original_state, export_file, status, debug_gallery, debug_panel]

    process_btn.click(
        fn=process_image,
        inputs=_process_inputs,
        outputs=_process_outputs,
        concurrency_limit=1,
    )

    process_btn_bottom.click(
        fn=process_image,
        inputs=_process_inputs,
        outputs=_process_outputs,
        concurrency_limit=1,
    )

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860)
