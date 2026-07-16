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
from retouch.io import imread_exif, imread_engine, EXPORT_RES_MAP, EXT_MAP
from retouch.lips import LIP_TINT_NAMES
from retouch.recipes import RECIPES
from retouch.params import recipe_to_params, PROCESSING_PARAMS, param_names, gui_values_to_engine_kwargs
from retouch.grading import list_available_presets
from retouch.style_library import list_styles, save_style_profile, learn_dataset_style
from retouch.batch_processor import BatchProcessor
from retouch.style import StyleProfile
from retouch.look_extractor import LookExtractor
from retouch.recipe_cookbook import search_recipes, list_recipes, list_categories
from retouch.lut import get_registry

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

    return tuple(d[k] for k in RECIPE_OUTPUT_KEYS)


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
        _logger.exception("Failed to save style: %s", e)
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
        _logger.exception("Error during dataset learning: %s", e)
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
        _logger.exception("Batch processing failed: %s", e)
        gr.Warning(f"Batch processing failed: {e}")
        return None, None, f"Exception during batch processing: {e}"


# PROCESS_INPUT_KEYS — the ordered list of inputs the process_image() Gradio
# event handler expects.  Generated from PROCESSING_PARAMS (in declaration
# order) so the slider order stays in lock-step with the spec, plus the
# fixed-prefix transport / session keys at the end.
PROCESS_INPUT_KEYS = (
    ["img_paths", "recipe"]
    + [n for n in param_names() if n not in ("color_transfer_intensity", "freckle_preserve_mask")]
    + [
        "color_ref_img", "color_ref_strength",
        "show_compare", "fast",
        "export_fmt", "export_quality", "export_res",
        "quality_tier",
        "debug_mode",
        "look_params",
        "face_params",
        "face_params_json",
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


# ---------------------------------------------------------------------------
# F2: Session handlers
# ---------------------------------------------------------------------------

def save_session_handler(*args):
    """Save current params as a session JSON file for download."""
    from retouch.session import create_session_from_params
    import tempfile, os

    params = dict(zip(PROCESS_INPUT_KEYS, args))
    img_paths = params.get("img_paths")
    image_path = None
    if img_paths and isinstance(img_paths, list) and len(img_paths) > 0:
        if isinstance(img_paths[0], str):
            image_path = img_paths[0]
        elif hasattr(img_paths[0], "name"):
            image_path = img_paths[0].name

    recipe = params.get("recipe", "natural")
    session = create_session_from_params(params, recipe=recipe, image_path=image_path)

    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, f"{recipe or 'session'}.session.json")
    session.to_file(filepath)

    return gr.update(value=filepath, visible=True)


def load_session_handler(session_file, *current_args):
    """Load a session JSON file and return params as a tuple for Gradio."""
    from retouch.session import Session
    import logging

    if session_file is None:
        return current_args

    try:
        if hasattr(session_file, "name"):
            path = session_file.name
        else:
            path = str(session_file)
        session = Session.from_file(path)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load session: {e}")
        gr.Warning(f"Failed to load session: {e}")
        return current_args

    gr.Info(f"Loaded session (recipe: {session.recipe or 'unknown'})")

    result = list(current_args)
    for i, key in enumerate(PROCESS_INPUT_KEYS):
        if key in session.params:
            val = session.params[key]
            if val is not None:
                if i < len(result):
                    result[i] = val
    return tuple(result)


def push_undo_handler(*args, undo_stack=None):
    """Push current params onto the undo stack."""
    from retouch.session import UndoRedoStack

    params = dict(zip(PROCESS_INPUT_KEYS, args))
    if undo_stack is None:
        undo_stack = UndoRedoStack()
    undo_stack.push(params)
    return undo_stack, gr.update(interactive=undo_stack.can_undo), gr.update(interactive=undo_stack.can_redo)


def undo_handler(undo_stack):
    """Undo: move cursor back and return params tuple."""
    if undo_stack is None:
        return tuple(None for _ in PROCESS_INPUT_KEYS), None, gr.update(interactive=False), gr.update(interactive=False)
    params = undo_stack.undo()
    if params is None:
        return tuple(None for _ in PROCESS_INPUT_KEYS), undo_stack, gr.update(interactive=False), gr.update(interactive=undo_stack.can_redo)
    result = tuple(params.get(k) for k in PROCESS_INPUT_KEYS)
    return result, undo_stack, gr.update(interactive=undo_stack.can_undo), gr.update(interactive=undo_stack.can_redo)


def redo_handler(undo_stack):
    """Redo: move cursor forward and return params tuple."""
    if undo_stack is None:
        return tuple(None for _ in PROCESS_INPUT_KEYS), None, gr.update(interactive=False), gr.update(interactive=False)
    params = undo_stack.redo()
    if params is None:
        return tuple(None for _ in PROCESS_INPUT_KEYS), undo_stack, gr.update(interactive=undo_stack.can_undo), gr.update(interactive=False)
    result = tuple(params.get(k) for k in PROCESS_INPUT_KEYS)
    return result, undo_stack, gr.update(interactive=undo_stack.can_undo), gr.update(interactive=undo_stack.can_redo)


def save_snapshot_handler(name, *args, snapshots=None):
    """Save a named snapshot of current params."""
    from retouch.session import Session, Snapshot

    if not name or not name.strip():
        gr.Warning("Please enter a snapshot name")
        return gr.update(), snapshots or {}

    params = dict(zip(PROCESS_INPUT_KEYS, args))
    recipe = params.get("recipe", "natural")
    session = Session(recipe=recipe, params=params)
    snap = Snapshot(name=name.strip(), session=session)

    if snapshots is None:
        snapshots = {}
    snapshots[name.strip()] = snap

    gr.Info(f"Snapshot '{name.strip()}' saved")
    return gr.update(choices=list(snapshots.keys())), snapshots


def compare_snapshot_handler(selected_name, *args, snapshots=None):
    """Compare current output with a saved snapshot."""
    if not selected_name or snapshots is None or selected_name not in snapshots:
        gr.Warning("Select a snapshot to compare")
        return None

    snap = snapshots[selected_name]
    gr.Info(f"Comparing with snapshot '{selected_name}' (recipe: {snap.session.recipe})")
    return snap.session.to_json()


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
    quality_tier = params.get("quality_tier")
    quality = "draft" if quality_tier and quality_tier.startswith("Draft") else "full"

    if not img_paths:
        return None, gr.update(visible=False), None, None, "Please upload an image first.", None, gr.update(visible=False), ""

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
        # Defensive check: only process if color_ref_img is a valid file path (not bool/invalid type)
        if isinstance(color_ref_img, dict):
            color_ref_img = color_ref_img.get("name") or color_ref_img.get("path")
        if color_ref_img and (isinstance(color_ref_img, (str, bytes)) or hasattr(color_ref_img, '__fspath__')):
            try:
                color_ref_bgr = imread_exif(color_ref_img)
            except (TypeError, FileNotFoundError) as e:
                _logger.warning("Failed to load color reference image: %s", e)

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
    qa_warnings = []
    qa_html = ""

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

    # F6: overlay extracted-look params over recipe defaults (look wins).
    look_params = params.get("look_params")
    if isinstance(look_params, dict):
        for k, v in look_params.items():
            if str(k).startswith("_"):
                continue
            engine_kwargs[k] = v

    # Per-face overrides: State dict wins; JSON textbox is advanced fallback.
    from retouch.face_params import coerce_face_params
    fp = params.get("face_params")
    if not fp:
        face_params_json = params.get("face_params_json") or ""
        if isinstance(face_params_json, str) and face_params_json.strip():
            try:
                import json as _json
                fp = _json.loads(face_params_json)
            except Exception as e:
                _logger.warning("face_params_json parse failed: %s", e)
                fp = None
    if fp:
        coerced = coerce_face_params(fp)
        if coerced:
            engine_kwargs["face_params"] = coerced

    for idx, path_item in enumerate(img_paths):
        try:
            curr_path = path_item
            if isinstance(path_item, dict):
                curr_path = path_item.get("name") or path_item.get("path")
            
            # T5: RAW via 16-bit path; JPEG/PNG unchanged (imread_engine)
            img_bgr = imread_engine(curr_path)
            original = (
                np.clip(img_bgr, 0, 255).astype(np.uint8)
                if img_bgr.dtype != np.uint8
                else img_bgr.copy()
            )

            engine_kwargs["debug_dir"] = (
                debug_dir if (first_result_rgb is None and first_combined is None) else None
            )

            engine_kwargs["quality"] = quality

            result = engine.process(img_bgr, **engine_kwargs)
            qa_warnings = getattr(result, 'qa', [])

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
            if export_fmt == "PNG-16":
                from retouch.io import write_image_16bit
                write_image_16bit(out_path, export_img, format="png")
            else:
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
        return None, gr.update(visible=False), None, None, "Error: No images were successfully processed.", None, gr.update(visible=False), qa_html

    preview = first_combined if show_compare else first_result_rgb
    debug_gallery = debug_images if debug_images else None
    debug_vis = gr.update(visible=bool(debug_images))

    qa_html = ""
    if qa_warnings:
        items = "".join(f'<li>⚠️ {w.message} (score: {w.score:.2f})</li>' for w in qa_warnings)
        qa_html = f'<div style="background:#fff3cd;border:1px solid #ffc107;padding:8px 12px;border-radius:6px;margin:8px 0;font-size:13px"><strong>Quality Warnings:</strong><ul style="margin:4px 0 0 16px;padding:0">{items}</ul></div>'

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
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_original, zip_path, f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s ✓", debug_gallery, debug_vis, qa_html
        return preview, gr.update(visible=False), first_original, zip_path, f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s ✓", debug_gallery, debug_vis, qa_html
    else:
        gr.Info(f"Done in {elapsed:.1f}s")

        if show_compare:
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_original, exported_paths[0], f"Done in {elapsed:.1f}s ✓", debug_gallery, debug_vis, qa_html
        return preview, gr.update(visible=False), first_original, exported_paths[0], f"Done in {elapsed:.1f}s ✓", debug_gallery, debug_vis, qa_html


def on_recipe_change(recipe):
    d = recipe_defaults(recipe)
    return tuple(d[k] for k in RECIPE_OUTPUT_KEYS)


def reset_skin_smoothing(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["smooth"], d["nose_smooth"], d["mid_reduction"], d["texture_opacity"], d["micro_restore"], d["pore_synthesis"], d["blemish"], d["skin_flatten"], d["skin_quantize"]

def reset_skin_tone(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["whiten"], d["whiten_tone"], d["equalize"], d["shadow_lift"], d["nose_restore"], d["skin_sss"], d["skin_unify"], d["skin_unify_hue"], d["auto_exposure"], d["white_costume_lift"], d["face_exposure"]

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
    return d["eye_enhance"], d["catchlight"], d["dark_circles"], d["undereye_darken_removal"], d["undereye_puffiness_reduction"], d["eye_sclera_brighten"], d["eye_iris_saturate"], d["eye_iris_hue_shift"], d["eye_iris_brightness"], d["teeth_whiten"], d["lip_enhance"], d["lip_tint"], d["lip_finish"], d["blush"], d["nose_blush"], d["under_eye_blush"]

def reset_face_reshaping(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["slimming"]

def reset_structure_effects(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["hair_enhance"], d["dodge_burn"], d["impact"], d["specular_bloom"], d["specular_bloom_tone"], d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["sharpen"], d["sharpen_radius"], d["glow"], d["skin_glow"], d["vignette"], d["subject_separation"]

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


def reset_body_skin(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["body_smooth"], d["body_equalize"], d["body_whiten"], d["body_match_face"], d["body_relight"], d["body_dodge_burn"], d["body_shadow_lift"]


def reset_lch(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["white_balance_kelvin"], d["white_balance_tint"], d["bw_channel_mixer_r"], d["bw_channel_mixer_g"], d["bw_channel_mixer_b"], d["negative_split_tone_shadow"], d["negative_split_tone_highlight"], d["hsl_hue_global"], d["hsl_sat_global"], d["hsl_lum_global"]


def _resolve_image_path(value):
    """Best-effort extraction of a filesystem path from a Gradio input value."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("name") or value.get("path")
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _resolve_image_path(value[0])
    if hasattr(value, "name"):
        return value.name
    return str(value)


def on_detect_faces(img_paths):
    """Detect faces → Gallery thumbs + face index choices. Clears face_params."""
    if not img_paths:
        return [], {}, gr.update(choices=[], value=None), "Upload an image first."
    path = _resolve_image_path(img_paths)
    if not path:
        return [], {}, gr.update(choices=[], value=None), "Could not resolve image path."
    try:
        img = imread_engine(path)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        faces = get_engine()._detector.detect(img)
    except Exception as e:
        _logger.warning("Detect faces failed: %s", e)
        return [], {}, gr.update(choices=[], value=None), f"Detect failed: {e}"
    if not faces:
        return [], {}, gr.update(choices=[], value=None), "No faces detected."
    thumbs = []
    choices = []
    for i, f in enumerate(faces):
        x, y, w, h = f.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        thumbs.append((rgb, f"Face {i}"))
        choices.append(str(i))
    
    suggested = get_engine().suggest_face_params(img, faces_data=faces)
    
    status = f"{len(choices)} face(s) detected.\n\nSuggested default recipes:\n"
    for k, v in suggested.items():
        status += f"- Face {k}: {v['recipe']}\n"
    status += "\nAssign a recipe per face (optional), then Process."

    return (
        thumbs,
        suggested,
        gr.update(choices=choices, value=choices[0] if choices else None),
        status,
    )


def on_apply_face_recipe(face_idx, recipe_name, face_params):
    """Merge {idx: {recipe: name}} into face_params State."""
    fp = dict(face_params or {})
    if face_idx is None or face_idx == "":
        return fp, "Select a face index first."
    if not recipe_name:
        return fp, "Select a recipe."
    try:
        i = int(face_idx)
    except (TypeError, ValueError):
        return fp, f"Bad face index: {face_idx!r}"
    entry = dict(fp.get(i) or fp.get(str(i)) or {})
    entry["recipe"] = recipe_name
    fp[i] = entry
    # drop string-key duplicate if any
    fp.pop(str(i), None)
    summary = ", ".join(f"{k}→{v.get('recipe', v)}" for k, v in sorted(fp.items()))
    return fp, f"Face {i} → {recipe_name}. Active: {summary}"


def on_clear_face_params():
    return {}, [], gr.update(choices=[], value=None), "Per-face overrides cleared."


def on_img_change_clear_faces():
    return {}, [], gr.update(choices=[], value=None), "Image changed — re-detect faces."


def on_extract_look(look_ref_file, img_input):
    """F6 — Extract an editable look from a reference image.

    Loads the uploaded reference, optionally pairs it with the main input
    image as the base, runs ``LookExtractor.extract``, strips ``_``-prefixed
    keys (internal transport params) and returns the cleaned ``engine_params``
    dict for storage in the hidden ``look_params`` State. The subsequent
    Process call overlays these params over the recipe defaults (look wins).
    Failures are reported without raising.
    """
    if look_ref_file is None:
        return {}, "Please upload a reference image first."

    ref_path = _resolve_image_path(look_ref_file)
    if not ref_path:
        return {}, "Could not resolve the reference image path."

    try:
        ref_bgr = imread_exif(ref_path)
    except (TypeError, FileNotFoundError, OSError) as e:
        _logger.warning("Extract Look: failed to load reference %s: %s", ref_path, e)
        return {}, f"Extract Look failed to load reference: {e}"

    base_bgr = None
    base_path = _resolve_image_path(img_input)
    if base_path:
        try:
            base_bgr = imread_exif(base_path)
        except (TypeError, FileNotFoundError, OSError) as e:
            _logger.warning("Extract Look: failed to load base %s: %s", base_path, e)
            base_bgr = None

    try:
        result = LookExtractor().extract(ref_bgr, base_bgr)
    except Exception as e:
        _logger.exception("LookExtractor failed: %s", e)
        return {}, f"Extract Look failed: {e}"

    engine_params = result.get("engine_params") or {}
    clean = {k: v for k, v in engine_params.items() if not str(k).startswith("_")}
    mode = result.get("mode", "unknown")
    return clean, f"Look extracted ({mode}) — {len(clean)} params. Click Process to apply."


def on_search_recipes(query, category=None):
    """T4 — Search/browse cookbook; optional category filter."""
    try:
        cat = category if category and category != "All" else None
        if query and str(query).strip():
            results = search_recipes(query.strip())
            if cat:
                results = [r for r in results if r.category == cat]
        else:
            results = list_recipes(cat)
    except Exception as e:
        _logger.exception("Recipe search failed: %s", e)
        return gr.update(choices=[]), f"Recipe search failed: {e}"

    choices = [r.name for r in results]
    msg = f"{len(results)} recipe(s)"
    if query:
        msg += f" matching '{query}'"
    if category and category != "All":
        msg += f" in {category}"
    return gr.update(choices=choices, value=None), msg + "."


def on_select_cookbook(name):
    """T4 — Apply a cookbook recipe selection to the active recipe Radio."""
    if not name:
        return gr.update(), ""
    return gr.update(value=name), f"Selected recipe: {name}"


def on_browse_category(category):
    """T4 — List recipes in category (or all)."""
    return on_search_recipes("", category)


def on_reload_luts():
    """Trigger a LUT registry reload (clears cache, re-scans luts dir)."""
    try:
        get_registry().reload()
        return "LUTs reloaded successfully."
    except Exception as e:
        _logger.warning("LUT reload failed: %s", e)
        return f"LUT reload failed: {e}"


def on_smart_process(img_paths, recipe, *args, prg=gr.Progress()):
    """F10 Smart Process — analyze the first image and auto-set sliders.

    Returns the new recipe value, the full slider tuple (matching
    ``_recipe_outputs``), and a status string with the explanation text.
    The caller (Gradio event) wires this to the recipe Radio + slider
    outputs + status Textbox so the UI updates in place. The user can then
    hit "Process Image(s)" to actually run the pipeline, or tweak the
    auto-filled sliders first.
    """
    from retouch.smart_default import SmartProcessor

    if not img_paths:
        gr.Warning("Please upload an image first.")
        return (gr.update(), *([gr.update()] * len(_recipe_outputs)),
                "Please upload an image first.", gr.update())

    curr_path = img_paths[0]
    if isinstance(curr_path, dict):
        curr_path = curr_path.get("name") or curr_path.get("path")

    try:
        img_bgr = imread_exif(curr_path)
    except (TypeError, FileNotFoundError, OSError) as e:
        _logger.warning("Smart Process: failed to load %s: %s", curr_path, e)
        gr.Warning(f"Failed to load image: {e}")
        return (gr.update(), *([gr.update()] * len(_recipe_outputs)),
                f"Failed to load image: {e}", gr.update())

    if img_bgr is None:
        gr.Warning("Could not read the image.")
        return (gr.update(), *([gr.update()] * len(_recipe_outputs)),
                "Could not read the image.", gr.update())

    gr.Info("🧠 Analyzing image...")
    sp = SmartProcessor()
    try:
        suggestion = sp.analyze_and_suggest(img_bgr)
    except ValueError as e:
        _logger.exception("Smart Process analysis failed: %s", e)
        gr.Warning(f"Analysis failed: {e}")
        return (gr.update(), *([gr.update()] * len(_recipe_outputs)),
                f"Analysis failed: {e}", gr.update())

    # Start from the suggested recipe's defaults, then layer the suggestion
    # overrides on top. This produces the full slider tuple that
    # on_recipe_change would return, with the smart overrides applied.
    d = recipe_defaults(suggestion.recipe)
    for k, v in suggestion.params.items():
        if k in d:
            d[k] = v

    # Build the slider output tuple in RECIPE_OUTPUT_KEYS order (key-based,
    # so it can never drift from the canonical output contract).
    slider_outputs = tuple(d[k] for k in RECIPE_OUTPUT_KEYS)

    explanation_html = _format_smart_explanations(suggestion)
    status_msg = f"🧠 Smart suggestion applied ({suggestion.recipe}). Click Process to run."
    gr.Info(f"Smart suggestion: {suggestion.recipe} ({len(suggestion.params)} overrides)")

    return (suggestion.recipe, *slider_outputs, status_msg, explanation_html)


def _format_smart_explanations(suggestion) -> str:
    """Format SmartSuggestion.explanations as an HTML readout for the GUI."""
    if not suggestion.explanations:
        return ""
    items = "".join(f"<li>{e}</li>" for e in suggestion.explanations)
    return (
        '<div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.25);'
        'padding:8px 12px;border-radius:6px;margin:8px 0;font-size:13px">'
        f'<strong>🧠 Smart Analysis — recipe: {suggestion.recipe}</strong>'
        f'<ul style="margin:4px 0 0 16px;padding:0">{items}</ul>'
        '</div>'
    )



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

                        with gr.Accordion("📖 Recipe Cookbook", open=False):
                            with gr.Row():
                                cookbook_category = gr.Dropdown(
                                    label="Category",
                                    choices=["All"] + list_categories(),
                                    value="All",
                                    interactive=True,
                                    scale=1,
                                )
                                cookbook_search = gr.Textbox(
                                    label="Search",
                                    placeholder="e.g. cosplay, portrait, fuji",
                                    scale=2,
                                )
                            with gr.Row():
                                cookbook_search_btn = gr.Button(
                                    "Search / Browse", variant="secondary", size="sm",
                                    elem_classes=["secondary-btn"],
                                )
                            cookbook_dropdown = gr.Dropdown(
                                label="Cookbook Recipe",
                                choices=[],
                                interactive=True,
                                value=None,
                                allow_custom_value=False,
                                info="Browse by category or search, then select to load into Base Recipe",
                            )
                            cookbook_status = gr.Markdown(
                                "Open accordion → pick category or search → select recipe."
                            )

                        show_compare = gr.Checkbox(label="Show side-by-side comparison screen", value=True, info="Split view: original | separator | retouched result")

                        with gr.Row():
                            fast = gr.Checkbox(label="Fast Preview (Recommended)", value=True,
                                               info="Process at half resolution for preview; final export always uses full quality")
                        
                        with gr.Row():
                            process_btn = gr.Button("Process Image(s) ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])
                            smart_process_btn = gr.Button("🧠 Smart Process", variant="primary", size="lg", elem_classes=["primary-btn"])
                            reset_btn = gr.Button("Reload Recipe Defaults 🔄", variant="secondary", size="lg", elem_classes=["secondary-btn"], elem_id="reset-btn")

                        with gr.Accordion("💾 Session & History", open=False):
                            with gr.Row():
                                save_session_btn = gr.Button("Save Session", variant="secondary", size="sm")
                                load_session_file = gr.File(label="Load Session", file_types=[".json"], file_count="single")
                                undo_btn = gr.Button("↩ Undo", variant="secondary", size="sm")
                                redo_btn = gr.Button("↻ Redo", variant="secondary", size="sm")
                            with gr.Row():
                                snapshot_name = gr.Textbox(label="Snapshot Name", placeholder="e.g. 'warm tone'", scale=3)
                                save_snapshot_btn = gr.Button("📸 Save Snapshot", variant="secondary", size="sm", scale=1)
                                snapshot_dropdown = gr.Dropdown(label="Snapshots", choices=[], scale=3)
                                compare_snapshot_btn = gr.Button("Compare", variant="secondary", size="sm", scale=1)
                            session_download = gr.File(label="Download Session", visible=False)
                            _undo_stack_state = gr.State(value=None)
                            _snapshot_state = gr.State(value={})

                    with gr.Group():
                        gr.Markdown("### ⚙️ Export Settings")
                        with gr.Row():
                            export_fmt = gr.Radio(choices=["JPEG", "PNG", "PNG-16", "WebP"], value="JPEG", label="Format", interactive=True, info="PNG-16 = 16-bit (higher precision, larger file)")
                            export_quality = gr.Slider(10, 100, 95, step=1, label="Compression Quality", info="For JPEG/WebP formats")
                        export_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Resize / Limit Resolution", interactive=True,
                            info="Downscales image if it exceeds target dimension while maintaining aspect ratio"
                        )
                        quality_tier = gr.Radio(choices=["Full (native face crops)", "Draft (proxy, fast)"], value="Full (native face crops)", label="Processing Quality", interactive=True, info="Full: faces processed at native resolution (F8.2). Draft: legacy proxy path for fast batch contact sheets.")

                # Column 2: Workspace Canvas (Center)
                with gr.Column(scale=4, elem_classes=["viewer-panel"]):
                    with gr.Group():
                        gr.Markdown("### 🖼️ Preview Canvas")
                        img_output = gr.Image(height=600, show_label=False, elem_id="retouch-output")
                        compare_viewer = gr.HTML(visible=False, elem_id="retouch-compare")
                        _original_state = gr.State(value=None)
                        # Hidden state variables for newly-added parameters (skin_hue_unify, skin_chroma_even)
                        # These maintain alignment with PROCESS_INPUT_KEYS but don't have visible UI yet.
                        _skin_hue_unify_state = gr.State(value=0)
                        _skin_chroma_even_state = gr.State(value=0)
                        _blotch_reduction_state = gr.State(value=0.0)
                        _specular_finish_state = gr.State(value="matte")
                        _specular_finish_strength_state = gr.State(value=0.5)
                        _specular_recolor_state = gr.State(value=0.0)
                        _albedo_even_state = gr.State(value=0.0)
                        _makeup_coverage_even_state = gr.State(value=0.0)
                        _makeup_cake_reduce_state = gr.State(value=0.0)
                        _hemoglobin_smooth_state = gr.State(value=0.0)
                        # mole_protect is a visible slider (below freckle_removal); no State
                        with gr.Accordion("👥 Per-face recipes", open=False):
                            detect_faces_btn = gr.Button("Detect Faces", size="sm", variant="secondary")
                            face_gallery = gr.Gallery(
                                label="Detected faces (detection order)",
                                columns=4, height=140, object_fit="cover",
                            )
                            with gr.Row():
                                face_select = gr.Dropdown(
                                    label="Face index", choices=[], interactive=True, scale=1,
                                )
                                face_recipe = gr.Dropdown(
                                    label="Recipe for face",
                                    choices=RECIPE_NAMES, value=None, scale=2,
                                )
                                apply_face_btn = gr.Button("Apply to face", size="sm", scale=1)
                            clear_faces_btn = gr.Button("Clear per-face", size="sm")
                            face_params_status = gr.Markdown("")
                            _face_params_state = gr.State(value={})
                            face_params_json = gr.Textbox(
                                label="Advanced: face_params JSON",
                                placeholder='{"0": {"recipe": "cosplay", "smooth": 70}}',
                                lines=2,
                                info="Optional. State from picker wins if set. Engine units 0–100.",
                            )
                        _vein_attenuate_state = gr.State(value=0.0)
                        _gamut_compress_state = gr.State(value=True)
                        _saturation_mode_state = gr.State(value="additive")
                        _reshape_eye_size_state = gr.State(value=0.0)
                        _reshape_eye_distance_state = gr.State(value=0.0)
                        _reshape_nose_width_state = gr.State(value=0.0)
                        _reshape_nose_length_state = gr.State(value=0.0)
                        _reshape_jaw_width_state = gr.State(value=0.0)
                        _reshape_chin_length_state = gr.State(value=0.0)
                        _reshape_mouth_size_state = gr.State(value=0.0)
                        _reshape_smile_state = gr.State(value=0.0)
                        _reshape_forehead_state = gr.State(value=0.0)
                        _hair_deglare_state = gr.State(value=0.0)
                        _hair_ring_position_state = gr.State(value=0.5)
                        _hair_ring_tint_state = gr.State(value=0.0)
                        _hair_remove_flyaways_state = gr.State(value=0.0)
                        _film_enable_state = gr.State(value=False)
                        _film_strength_state = gr.State(value=0.0)
                        _film_toe_r_state = gr.State(value=0.0)
                        _film_toe_g_state = gr.State(value=0.0)
                        _film_toe_b_state = gr.State(value=0.0)
                        _film_shoulder_r_state = gr.State(value=0.0)
                        _film_shoulder_g_state = gr.State(value=0.0)
                        _film_shoulder_b_state = gr.State(value=0.0)
                        _film_midpoint_state = gr.State(value=0.5)
                        _film_gamma_state = gr.State(value=1.0)
                        _film_crosstalk_cy_mg_state = gr.State(value=0.0)
                        _film_crosstalk_cy_ye_state = gr.State(value=0.0)
                        _film_crosstalk_mg_ye_state = gr.State(value=0.0)
                        _film_tonemap_strength_state = gr.State(value=0.0)
                        _film_tonemap_toe_state = gr.State(value=0.1)
                        _film_tonemap_shoulder_state = gr.State(value=0.1)
                        _film_skew_state = gr.State(value=0.0)
                        _background_harmonize_state = gr.State(value=0.0)
                        _background_harmonize_mode_state = gr.State(value="split")
                        _background_blur_state = gr.State(value=0.0)
                        _background_desaturation_state = gr.State(value=0.0)
                        _light_wrap_state = gr.State(value=0.0)
                        _blue_shadow_grade_state = gr.State(value=0.0)
                        _cyan_midtone_grade_state = gr.State(value=0.0)
                        _subject_sharpen_state = gr.State(value=0.0)
                        _matte_black_state = gr.State(value=0.0)
                        _ai_denoise_state = gr.State(value=0.0)
                        _ai_sr_scale_state = gr.State(value=1)
                        _mv2_eyeshadow_state = gr.State(value=0.0)
                        _mv2_eyeshadow_color_state = gr.State(value="brown")
                        _mv2_eyeshadow_style_state = gr.State(value="natural")
                        _mv2_eyeliner_state = gr.State(value=0.0)
                        _mv2_eyeliner_color_state = gr.State(value="black")
                        _mv2_eyeliner_style_state = gr.State(value="classic")
                        _mv2_contour_state = gr.State(value=0.0)
                        _mv2_brows_state = gr.State(value=0.0)
                        _mv2_brows_color_state = gr.State(value="brown")
                        _mv2_ombre_state = gr.State(value=0.0)
                        _mv2_ombre_color1_state = gr.State(value="red")
                        _mv2_ombre_color2_state = gr.State(value="pink")
                        # Hidden states for params present in params.py but with no visible
                        # slider yet (wiring-debt alignment: every PROCESSING_PARAMS name
                        # must appear in _process_inputs in param_names() order).
                        _wrinkle_soften_forehead_state = gr.State(value=0)
                        _wrinkle_soften_nasolabial_state = gr.State(value=0)
                        _wrinkle_soften_neck_state = gr.State(value=0)
                        _eye_sclera_vessel_remove_state = gr.State(value=0)
                        _backdrop_cleanup_state = gr.State(value=0)
                        _fabric_wrinkle_smooth_state = gr.State(value=0.0)
                        _reshape_jaw_width_l_state = gr.State(value=0.0)
                        _reshape_jaw_width_r_state = gr.State(value=0.0)
                        _reshape_nose_width_l_state = gr.State(value=0.0)
                        _reshape_nose_width_r_state = gr.State(value=0.0)
                        _reshape_eye_size_l_state = gr.State(value=0.0)
                        _reshape_eye_size_r_state = gr.State(value=0.0)
                        _reshape_neck_width_state = gr.State(value=0.0)
                        _reshape_neck_length_state = gr.State(value=0.0)
                        _neural_stray_hair_boost_state = gr.State(value=0)
                        _neural_defect_boost_state = gr.State(value=0)
                        _lens_blur_state = gr.State(value=0.0)
                        _hsl_hue_red_state = gr.State(value=0.0)
                        _hsl_sat_red_state = gr.State(value=0.0)
                        _hsl_lum_red_state = gr.State(value=0.0)
                        _hsl_hue_orange_state = gr.State(value=0.0)
                        _hsl_sat_orange_state = gr.State(value=0.0)
                        _hsl_lum_orange_state = gr.State(value=0.0)
                        _hsl_hue_yellow_state = gr.State(value=0.0)
                        _hsl_sat_yellow_state = gr.State(value=0.0)
                        _hsl_lum_yellow_state = gr.State(value=0.0)
                        _hsl_hue_green_state = gr.State(value=0.0)
                        _hsl_sat_green_state = gr.State(value=0.0)
                        _hsl_lum_green_state = gr.State(value=0.0)
                        _hsl_hue_cyan_state = gr.State(value=0.0)
                        _hsl_sat_cyan_state = gr.State(value=0.0)
                        _hsl_lum_cyan_state = gr.State(value=0.0)
                        _hsl_hue_blue_state = gr.State(value=0.0)
                        _hsl_sat_blue_state = gr.State(value=0.0)
                        _hsl_lum_blue_state = gr.State(value=0.0)
                        _hsl_hue_purple_state = gr.State(value=0.0)
                        _hsl_sat_purple_state = gr.State(value=0.0)
                        _hsl_lum_purple_state = gr.State(value=0.0)
                        _hsl_hue_magenta_state = gr.State(value=0.0)
                        _hsl_sat_magenta_state = gr.State(value=0.0)
                        _hsl_lum_magenta_state = gr.State(value=0.0)
                        _calibration_red_hue_state = gr.State(value=0.0)
                        _calibration_red_sat_state = gr.State(value=0.0)
                        _calibration_red_lum_state = gr.State(value=0.0)
                        _calibration_green_hue_state = gr.State(value=0.0)
                        _calibration_green_sat_state = gr.State(value=0.0)
                        _calibration_green_lum_state = gr.State(value=0.0)
                        _calibration_blue_hue_state = gr.State(value=0.0)
                        _calibration_blue_sat_state = gr.State(value=0.0)
                        _calibration_blue_lum_state = gr.State(value=0.0)
                        _look_params_state = gr.State(value={})
                        status = gr.Textbox(label="Status", interactive=False, placeholder="Upload an image and click Process to start...")
                        smart_analysis_html = gr.HTML(visible=True)
                        qa_status = gr.HTML(visible=True)
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
                            smooth_engine = gr.Dropdown(choices=["guided", "bilateral", "anisotropic"], value="guided", label="Smoothing Engine", info="guided=isotropic (fast); anisotropic=orientation-aware (preserves wrinkle direction)")
                            regional_modulation = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Region-Aware Modulation", info="Per-region smoothing strength (0=off, 1=full modulation)")
                            mid_reduction = gr.Slider(0.0, 1.0, 0.45, step=0.05, label="Mid Frequency Reduction", info="Target mid-level skin blemishes while preserving high-frequency pores")
                            texture_opacity = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Texture Opacity", info="Control original pore structure opacity overlay")
                            micro_restore = gr.Slider(0, 50, 20, step=1, label="Micro-Texture Restore", info="Re-inject dimensional micro-contrast in cheek/nose/under-eye zones after smoothing (0 = off, 25 = subtle, 50 = strong)")
                            _micro_dodge_burn_state = gr.State(value=0)
                            _redness_even_state = gr.State(value=0)
                            _whiten_hue_stable_state = gr.State(value=0)
                            pore_synthesis = gr.Slider(0, 100, 0, step=1, label="Pore Synthesis", info="Add micro-texture/synthesized pores to prevent artificial plastic skin")
                            blemish = gr.Slider(0, 100, 30, step=1, label="Blemish Removal", info="AI blemish detection and inpainting for acne/spots")
                            freckle_removal = gr.Slider(0, 100, 0, step=1, label="Freckle Removal", info="Remove freckles while preserving beauty marks (0=off)")
                            mole_protect = gr.Slider(
                                0.0, 1.0, 0.0, step=0.05,
                                label="Mole / Beauty-Mark Protect",
                                info="R10: protect compact melanin spots from blemish+freckle heals (0=off, 1=full). Classical, no paid corpus.",
                            )
                            skin_flatten = gr.Slider(0, 100, 0, step=1, label="Skin Flatten (Anime)", info="Edge-preserving cel flatten for anime-style shading · 0=off, 80=aggressive")
                            skin_quantize = gr.Slider(0, 100, 0, step=1, label="Tone Quantize (Anime)", info="Cel shading colour bands on skin · 0=off, 60=dramatic bands")

                        with gr.Accordion("🎨 Skin Tone", open=False):
                            reset_skin_tone_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            whiten = gr.Slider(0, 100, 10, step=1, label="Whitening", info="Luminance boost and porcelain skin color match")
                            whiten_tone = gr.Dropdown(choices=WHITEN_TONE_CHOICES, value="rosy", label="Whitening Tone", interactive=True, info="Tone direction: rosy (warm pink), porcelain (cool neutral), neutral")
                            equalize = gr.Slider(0, 100, 20, step=1, label="Equalize", info="Even out skin redness and regional color inconsistencies")
                            shadow_lift = gr.Slider(0, 100, 0, step=1, label="Shadow Lift", info="Brighten small localized face shadows relative to local neighborhood")
                            nose_restore = gr.Slider(0, 100, 0, step=1, label="Nose Restore", info="Blend original (pre-retouch) nose pixels back in, to preserve natural nose shading")
                            skin_sss = gr.Slider(0, 100, 0, step=1, label="Subsurface Scatter", info="Game-render skin translucency: red-weighted shading diffusion + warm shadow terminators (pores stay crisp)")
                            skin_unify = gr.Slider(0, 100, 0, step=1, label="Skin Hue Unify (Anime)", info="Pull skin hues toward a single cel color · 0=off, 60=strong unified look")
                            skin_unify_hue = gr.Slider(-1.0, 360.0, -1.0, step=1.0, label="Target Hue (Anime)", info="Target skin hue angle · -1=auto (detect from face), 0=red, 50=orange, 180=cyan")
                            auto_exposure = gr.Checkbox(label="Auto Exposure Correction", value=False, info="Automatically correct under/over-exposed images before processing")
                            white_costume_lift = gr.Checkbox(label="White Costume Lift", value=False, info="Selectively boost bright clothing to create separation")
                            face_exposure = gr.Slider(0, 100, 0, step=1, label="Face Exposure Lift", info="Brighten/darken the exposed face relative to the body (skin.face_exposure) · 0 = off")

                        with gr.Accordion("🦵 Body Skin", open=False):
                            reset_body_skin_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            body_smooth = gr.Slider(0, 100, 0, step=1, label="Body Smooth", info="Smoothing for arms, legs, décolletage · milder curve than face to preserve texture")
                            body_equalize = gr.Slider(0, 100, 0, step=1, label="Body Equalize", info="Even out tone in body skin regions · tone harmonization at body scale")
                            body_whiten = gr.Slider(0, 100, 0, step=1, label="Body Whiten", info="Lighten body skin to match face whitening treatment")
                            body_match_face = gr.Slider(0, 100, 0, step=1, label="Body Match Face", info="Pull body skin L/a/b toward retouched face skin color · bounded ±8L ±6a/b")
                            body_relight = gr.Slider(0, 100, 0, step=1, label="Body Relight", info="Landmark-free directional shading on exposed body skin · matches face relight intensity")
                            body_dodge_burn = gr.Slider(0, 100, 0, step=1, label="Body Dodge & Burn", info="Local-contrast sculpting on body skin (CLAHE-based highlight/shadow)")
                            body_shadow_lift = gr.Slider(0, 100, 0, step=1, label="Body Shadow Lift", info="Brighten small localized shadows on body skin")

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
                            sculpt = gr.Slider(0, 100, 0, step=1, label="Facial Sculpting", info="Shape reflectance: deepen cheekbones, nose ridge, and jawline via low-band shading")
                            shine_removal = gr.Slider(0, 100, 0, step=1, label="Shine Removal", info="Remove oily/sweaty shine: compress specular highlights and reconstruct chroma")
                            wrinkle_soften = gr.Slider(0, 100, 0, step=1, label="Wrinkle & Line Softening", info="Reduce nasolabial folds, forehead lines, and crow's feet via ridge-aware attenuation")
                            texture_transplant = gr.Slider(0, 100, 0, step=1, label="Texture Transplant", info="Clone pore texture from clean skin regions to over-smoothed/inpainted zones for realistic texture")

                        with gr.Accordion("👁️ Eyes & Lips", open=False):
                            reset_eyes_lips_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            eye_enhance = gr.Slider(0, 100, 5, step=1, label="Eye Enhance", info="Boost eye clarity, iris reflection details, and whites brightness")
                            catchlight = gr.Slider(0, 100, 0, step=1, label="Catchlight Boost", info="Amplify existing catchlight highlights in the iris (0 = follow Eye Enhance)")
                            dark_circles = gr.Slider(0, 100, 0, step=1, label="Dark Circle Repair", info="Under-eye dark circle detection and repair")
                            undereye_shadow_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Under-Eye Shadow Smooth", info="Soften under-eye shadows conservatively (0=off)")
                            undereye_darken_removal = gr.Slider(0, 100, 0, step=1, label="Under-Eye Darken Removal", info="Lift under-eye darkening / discoloration (0=off)")
                            undereye_puffiness_reduction = gr.Slider(0, 100, 0, step=1, label="Under-Eye Puffiness Reduction", info="Reduce under-eye puffiness / bag volume (0=off)")
                            eye_sclera_brighten = gr.Slider(0, 100, 0, step=1, label="Sclera Brighten", info="Whiten/brighten the eye whites (sclera) for a clean look")
                            eye_iris_saturate = gr.Slider(0, 100, 0, step=1, label="Iris Saturate", info="Deepen iris color saturation")
                            eye_iris_brightness = gr.Slider(0, 100, 0, step=1, label="Iris Brightness", info="Brighten iris detail and reflection")
                            eye_iris_hue_shift = gr.Slider(-30, 30, 0, step=1, label="Iris Hue Shift", info="Rotate iris hue for colored-contact effects (-30..30°)")
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
                            skin_glow = gr.Slider(0, 100, 0, step=1, label="Skin Light-Wrap (Anime)", info="Skin-scoped diffusion glow / light-wrap for anime cel blending · 0=off, 30=visible halo")
                            vignette = gr.Slider(0, 100, 0, step=1, label="Vignette", info="Darken image corners for a focused portrait look")
                            fade_toe = gr.Slider(0, 100, 0, step=1, label="Fade Toe", info="Lift shadows while preserving hue (L-only LAB fade for 透明感)")
                            highlight_drift = gr.Slider(0, 100, 0, step=1, label="Highlight Drift", info="Bounded cyan hue rotation in highlights with skin protection")
                            airy_haze = gr.Slider(0, 100, 0, step=1, label="Airy Haze", info="L-threshold-scoped atmospheric glow for 空気感 effect")
                            clarity_split_neg = gr.Slider(0, 100, 0, step=1, label="Clarity Split (Form)", info="Reduce form-band local contrast for soft look")
                            clarity_split_pos = gr.Slider(0, 100, 0, step=1, label="Clarity Split (Texture)", info="Boost texture-band micro-contrast for detail")
                            subject_separation = gr.Slider(0, 100, 0, step=1, label="Subject-Background Separation", info="Brighten subject / darken background using person segmentation mask")

                        with gr.Accordion("🎭 Cosplay Moat (A3)", open=False):
                            gr.Markdown("Cosplay-specific skin / wardrobe continuity (wig lace blend, stockings smooth, cross-shot consistency).")
                            cosplay_wig_lace_blend = gr.Slider(0, 100, 0, step=1, label="Wig Lace Blend", info="Fade wig lace edge into forehead skin")
                            cosplay_stockings_smooth = gr.Slider(0, 100, 0, step=1, label="Stockings Smooth", info="Smooth hosiery / stocking texture")
                            cosplay_consistency_strength = gr.Slider(0, 100, 0, step=1, label="Consistency Strength", info="Cross-shot lighting / white-balance continuity for a cosplay set")

                        with gr.Accordion("🦵 Body Reshape (T3)", open=False):
                            gr.Markdown("Landmark-driven body reshape via MediaPipe Pose (±15% segment displacement at ±100). 50 = no change.")
                            body_reshape_arm_length = gr.Slider(0, 100, 50, step=1, label="Arm Length", info="0 = shorter, 100 = longer arms")
                            body_reshape_leg_length = gr.Slider(0, 100, 50, step=1, label="Leg Length", info="0 = shorter, 100 = longer legs")
                            body_reshape_torso_width = gr.Slider(0, 100, 50, step=1, label="Torso Width", info="0 = narrower, 100 = wider torso")
                            body_reshape_shoulder_width = gr.Slider(0, 100, 50, step=1, label="Shoulder Width", info="0 = narrower, 100 = wider shoulders")
                            body_reshape_hip_width = gr.Slider(0, 100, 50, step=1, label="Hip Width", info="0 = narrower, 100 = wider hips")
                            auto_body_reshape = gr.Slider(0, 100, 0, step=1, label="Auto Body Reshape", info="Automatic proportional reshape strength (0 = off)")

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

                        with gr.Accordion("🎨 LCH Color Tools", open=False):
                            reset_lch_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("**White Balance**")
                            white_balance_kelvin = gr.Slider(2000, 12000, 6500, step=100, label="Temperature (K)", info="2000=warm candlelight, 6500=neutral daylight, 12000=cool shade")
                            white_balance_tint = gr.Slider(-100, 100, 0, step=1, label="Tint", info="Negative=green correction, positive=magenta correction")
                            gr.Markdown("**B&W Channel Mixer**")
                            bw_channel_mixer_r = gr.Slider(-100, 200, 30, step=1, label="Red Weight", info="Red channel weight for B&W conversion")
                            bw_channel_mixer_g = gr.Slider(-100, 200, 59, step=1, label="Green Weight", info="Green channel weight for B&W conversion")
                            bw_channel_mixer_b = gr.Slider(-100, 200, 11, step=1, label="Blue Weight", info="Blue channel weight for B&W conversion")
                            gr.Markdown("**Negative Split Tone**")
                            negative_split_tone_shadow = gr.Slider(0, 100, 0, step=1, label="Shadow Desaturation", info="Fade shadows toward grayscale")
                            negative_split_tone_highlight = gr.Slider(0, 100, 0, step=1, label="Highlight Desaturation", info="Fade highlights toward grayscale")
                            gr.Markdown("**Master HSL**")
                            hsl_hue_global = gr.Slider(-100, 100, 0, step=1, label="Hue Shift", info="Global hue rotation in LCH space")
                            hsl_sat_global = gr.Slider(-100, 100, 0, step=1, label="Saturation", info="Global perceptual saturation ±100%")
                            hsl_lum_global = gr.Slider(-100, 100, 0, step=1, label="Luminance", info="Global L* lightness ±100")

                        with gr.Accordion("🔮 Color Transfer", open=False):
                            reset_color_transfer_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("Upload a reference image to match its color tone using CDF-based histogram transfer")
                            color_ref_img = gr.Image(type="filepath", label="Reference Image", show_label=True, height=160)
                            color_ref_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Transfer Strength", info="Mix ratio between original grade and matched reference grade (1.0 = full transfer, 0.0 = no transfer)")

                        with gr.Accordion("🎨 Look Extractor (F6)", open=False):
                            gr.Markdown("Upload a reference image to reverse-engineer an editable tone/color look. The extracted params are applied on the next Process (overriding recipe defaults).")
                            look_ref_file = gr.File(label="Reference Image (look source)", file_types=["image"], file_count="single")
                            look_extract_btn = gr.Button("✨ Extract Look", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                            look_status = gr.Markdown("")

                        with gr.Accordion("🎞️ LUT Library", open=False):
                            gr.Markdown("Hot-reload the 3D LUT registry after adding/removing `.cube` files in the luts directory.")
                            reload_luts_btn = gr.Button("🔄 Reload LUTs", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                            reload_luts_status = gr.Markdown("")

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
    # RECIPE_OUTPUT_KEYS -- canonical ordered list of UI-output param names.
    # It is the single ordering contract for every producer that fills the
    # recipe / custom-style / smart-process slider tuple.  Producers must emit
    # values keyed by THIS list (never a hand-ordered tuple), so a newly added
    # param cannot silently shift every later slider into the wrong value
    # (the historical "brightness lands in the blush slider" footgun).
    RECIPE_OUTPUT_KEYS = (
        "smooth",
 "mid_reduction",
 "texture_opacity",
 "pore_synthesis",
 "nose_smooth",
        "regional_modulation",
 "smooth_engine",
 "undereye_shadow_strength",
 "freckle_removal",
 "micro_restore",
        "whiten",
 "equalize",
 "blemish",
 "whiten_tone",
 "nose_blush",
        "under_eye_blush",
 "white_costume_lift",
 "body_smooth",
 "body_equalize",
 "body_whiten",
        "body_match_face",
 "dodge_burn",
 "relight",
 "relight_azimuth",
 "relight_elevation",
        "sculpt",
 "shine_removal",
 "wrinkle_soften",
 "specular_bloom",
 "specular_bloom_tone",
        "skin_flatten",
 "skin_quantize",
 "skin_unify",
 "skin_unify_hue",
 "skin_glow",
 "skin_sss",
        "eye_enhance",
 "catchlight",
 "dark_circles",
 "undereye_darken_removal",
 "undereye_puffiness_reduction",
        "eye_sclera_brighten",
 "eye_iris_saturate",
 "eye_iris_hue_shift",
 "eye_iris_brightness",
 "teeth_whiten",
        "lip_enhance",
 "lip_tint",
 "lip_finish",
 "blush",
 "slimming",
        "hair_enhance",
 "contrast",
 "brightness",
 "highlights",
 "shadows",
        "whites",
 "blacks",
 "clarity",
 "vibrance",
 "saturation",
        "auto_exposure",
 "bloom",
 "bloom_threshold",
 "bloom_softness",
 "glow",
        "vignette",
 "sharpen",
 "sharpen_radius",
 "fade_toe",
 "highlight_drift",
        "airy_haze",
 "clarity_split_neg",
 "clarity_split_pos",
 "subject_separation",
 "impact",
        "color_grade",
 "grade_intensity",
 "chromatic_aberration",
 "grain",
 "halation",
        "lut",
 "tonal_curve_strength",
 "skin_protect_strength",
 "grain_strength",
 "highlight_rolloff_strength",
        "shadow_hue",
 "shadow_sat",
 "midtone_hue",
 "midtone_sat",
 "highlight_hue",
        "highlight_sat",
 "white_balance_kelvin",
 "white_balance_tint",
 "bw_channel_mixer_r",
 "bw_channel_mixer_g",
        "bw_channel_mixer_b",
 "negative_split_tone_shadow",
 "negative_split_tone_highlight",
 "hsl_hue_global",
 "hsl_sat_global",
        "hsl_lum_global",
 "face_exposure",
 "cosplay_wig_lace_blend",
 "cosplay_stockings_smooth",
 "cosplay_consistency_strength",
        "body_reshape_arm_length",
 "body_reshape_leg_length",
 "body_reshape_torso_width",
 "body_reshape_shoulder_width",
 "body_reshape_hip_width",
        "auto_body_reshape",
        # Recipe-driven face/body skin sliders that are VISIBLE gr.Slider
        # components (not gr.State placeholders): they must be synced on recipe
        # change or the stale slider value stomps the recipe value in GUI
        # renders (CLI was already correct via build_context).  Appended so
        # every existing RECIPE_OUTPUT_KEYS index stays stable.
        "shadow_lift",
 "nose_restore",
 "mole_protect",
 "texture_transplant",
 "body_relight",
        "body_dodge_burn",
 "body_shadow_lift"
    )

    # Name -> Gradio component map for the recipe-output tuple.  Mirrors the
    # `_process_input_components` guard: every RECIPE_OUTPUT_KEYS name must map
    # to exactly one component and vice-versa, enforced at import time so a
    # missing/extra entry fails loudly instead of shifting slider values.
    _recipe_output_components = {
        "smooth": smooth,
 "mid_reduction": mid_reduction,
 "texture_opacity": texture_opacity,
        "pore_synthesis": pore_synthesis,
 "nose_smooth": nose_smooth,
 "regional_modulation": regional_modulation,
        "smooth_engine": smooth_engine,
 "undereye_shadow_strength": undereye_shadow_strength,
 "freckle_removal": freckle_removal,
        "micro_restore": micro_restore,
 "whiten": whiten,
 "equalize": equalize,
        "blemish": blemish,
 "whiten_tone": whiten_tone,
 "nose_blush": nose_blush,
        "under_eye_blush": under_eye_blush,
 "white_costume_lift": white_costume_lift,
 "body_smooth": body_smooth,
        "body_equalize": body_equalize,
 "body_whiten": body_whiten,
 "body_match_face": body_match_face,
        "dodge_burn": dodge_burn,
 "relight": relight,
 "relight_azimuth": relight_azimuth,
        "relight_elevation": relight_elevation,
 "sculpt": sculpt,
 "shine_removal": shine_removal,
        "wrinkle_soften": wrinkle_soften,
 "specular_bloom": specular_bloom,
 "specular_bloom_tone": specular_bloom_tone,
        "skin_flatten": skin_flatten,
 "skin_quantize": skin_quantize,
 "skin_unify": skin_unify,
        "skin_unify_hue": skin_unify_hue,
 "skin_glow": skin_glow,
 "skin_sss": skin_sss,
 "eye_enhance": eye_enhance,
        "catchlight": catchlight,
 "dark_circles": dark_circles,
 "undereye_darken_removal": undereye_darken_removal,
        "undereye_puffiness_reduction": undereye_puffiness_reduction,
 "eye_sclera_brighten": eye_sclera_brighten,
 "eye_iris_saturate": eye_iris_saturate,
        "eye_iris_hue_shift": eye_iris_hue_shift,
 "eye_iris_brightness": eye_iris_brightness,
 "teeth_whiten": teeth_whiten,
        "lip_enhance": lip_enhance,
 "lip_tint": lip_tint,
 "lip_finish": lip_finish,
        "blush": blush,
 "slimming": slimming,
 "hair_enhance": hair_enhance,
        "contrast": contrast,
 "brightness": brightness,
 "highlights": highlights,
        "shadows": shadows,
 "whites": whites,
 "blacks": blacks,
        "clarity": clarity,
 "vibrance": vibrance,
 "saturation": saturation,
        "auto_exposure": auto_exposure,
 "bloom": bloom,
 "bloom_threshold": bloom_threshold,
        "bloom_softness": bloom_softness,
 "glow": glow,
 "vignette": vignette,
        "sharpen": sharpen,
 "sharpen_radius": sharpen_radius,
 "fade_toe": fade_toe,
        "highlight_drift": highlight_drift,
 "airy_haze": airy_haze,
 "clarity_split_neg": clarity_split_neg,
        "clarity_split_pos": clarity_split_pos,
 "subject_separation": subject_separation,
 "impact": impact,
        "color_grade": color_grade,
 "grade_intensity": grade_intensity,
 "chromatic_aberration": chromatic_aberration,
        "grain": grain,
 "halation": halation,
 "lut": lut,
        "tonal_curve_strength": tonal_curve_strength,
 "skin_protect_strength": skin_protect_strength,
 "grain_strength": grain_strength,
        "highlight_rolloff_strength": highlight_rolloff_strength,
 "shadow_hue": shadow_hue,
 "shadow_sat": shadow_sat,
        "midtone_hue": midtone_hue,
 "midtone_sat": midtone_sat,
 "highlight_hue": highlight_hue,
        "highlight_sat": highlight_sat,
 "white_balance_kelvin": white_balance_kelvin,
 "white_balance_tint": white_balance_tint,
        "bw_channel_mixer_r": bw_channel_mixer_r,
 "bw_channel_mixer_g": bw_channel_mixer_g,
 "bw_channel_mixer_b": bw_channel_mixer_b,
        "negative_split_tone_shadow": negative_split_tone_shadow,
 "negative_split_tone_highlight": negative_split_tone_highlight,
 "hsl_hue_global": hsl_hue_global,
        "hsl_sat_global": hsl_sat_global,
 "hsl_lum_global": hsl_lum_global,
 "face_exposure": face_exposure,
        "cosplay_wig_lace_blend": cosplay_wig_lace_blend,
 "cosplay_stockings_smooth": cosplay_stockings_smooth,
 "cosplay_consistency_strength": cosplay_consistency_strength,
        "body_reshape_arm_length": body_reshape_arm_length,
 "body_reshape_leg_length": body_reshape_leg_length,
 "body_reshape_torso_width": body_reshape_torso_width,
        "body_reshape_shoulder_width": body_reshape_shoulder_width,
 "body_reshape_hip_width": body_reshape_hip_width,
 "auto_body_reshape": auto_body_reshape,
        # Visible recipe-driven skin sliders (see RECIPE_OUTPUT_KEYS note above).
        "shadow_lift": shadow_lift,
 "nose_restore": nose_restore,
 "mole_protect": mole_protect,
 "texture_transplant": texture_transplant,
 "body_relight": body_relight,
        "body_dodge_burn": body_dodge_burn,
 "body_shadow_lift": body_shadow_lift
    }
    _missing_outputs = set(RECIPE_OUTPUT_KEYS) - set(_recipe_output_components)
    _extra_outputs = set(_recipe_output_components) - set(RECIPE_OUTPUT_KEYS)
    if _missing_outputs or _extra_outputs:
        raise AssertionError(
            "_recipe_output_components drift vs RECIPE_OUTPUT_KEYS: "
            f"missing={sorted(_missing_outputs)} extra={sorted(_extra_outputs)}"
        )
    _recipe_outputs = [_recipe_output_components[k] for k in RECIPE_OUTPUT_KEYS]


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
        outputs=[smooth, nose_smooth, mid_reduction, texture_opacity, micro_restore, pore_synthesis, blemish, skin_flatten, skin_quantize]
    )

    reset_skin_tone_btn.click(
        fn=reset_skin_tone,
        inputs=[recipe],
        outputs=[whiten, whiten_tone, equalize, shadow_lift, nose_restore, skin_sss, skin_unify, skin_unify_hue, auto_exposure, white_costume_lift, face_exposure]
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
        outputs=[eye_enhance, catchlight, dark_circles, undereye_darken_removal, undereye_puffiness_reduction, eye_sclera_brighten, eye_iris_saturate, eye_iris_hue_shift, eye_iris_brightness, teeth_whiten, lip_enhance, lip_tint, lip_finish, blush, nose_blush, under_eye_blush]
    )

    reset_face_reshaping_btn.click(
        fn=reset_face_reshaping,
        inputs=[recipe],
        outputs=[slimming]
    )

    reset_structure_effects_btn.click(
        fn=reset_structure_effects,
        inputs=[recipe],
        outputs=[hair_enhance, dodge_burn, impact, specular_bloom, specular_bloom_tone, bloom, bloom_threshold, bloom_softness, sharpen, sharpen_radius, glow, skin_glow, vignette, subject_separation]
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

    reset_body_skin_btn.click(
        fn=reset_body_skin,
        inputs=[recipe],
        outputs=[body_smooth, body_equalize, body_whiten, body_match_face, body_relight, body_dodge_burn, body_shadow_lift]
    )

    reset_lch_btn.click(
        fn=reset_lch,
        inputs=[recipe],
        outputs=[white_balance_kelvin, white_balance_tint, bw_channel_mixer_r, bw_channel_mixer_g, bw_channel_mixer_b, negative_split_tone_shadow, negative_split_tone_highlight, hsl_hue_global, hsl_sat_global, hsl_lum_global]
    )

    # F6: Look Extractor wiring
    look_extract_btn.click(
        fn=on_extract_look,
        inputs=[look_ref_file, img_input],
        outputs=[_look_params_state, look_status],
    )

    # Per-face recipe picker
    detect_faces_btn.click(
        fn=on_detect_faces,
        inputs=[img_input],
        outputs=[face_gallery, _face_params_state, face_select, face_params_status],
    )
    apply_face_btn.click(
        fn=on_apply_face_recipe,
        inputs=[face_select, face_recipe, _face_params_state],
        outputs=[_face_params_state, face_params_status],
    )
    clear_faces_btn.click(
        fn=on_clear_face_params,
        inputs=[],
        outputs=[_face_params_state, face_gallery, face_select, face_params_status],
    )
    img_input.change(
        fn=on_img_change_clear_faces,
        inputs=[],
        outputs=[_face_params_state, face_gallery, face_select, face_params_status],
    )

    # T4: Recipe Cookbook wiring
    cookbook_search_btn.click(
        fn=on_search_recipes,
        inputs=[cookbook_search, cookbook_category],
        outputs=[cookbook_dropdown, cookbook_status],
    )
    cookbook_category.change(
        fn=on_browse_category,
        inputs=[cookbook_category],
        outputs=[cookbook_dropdown, cookbook_status],
    )
    cookbook_dropdown.change(
        fn=on_select_cookbook,
        inputs=[cookbook_dropdown],
        outputs=[recipe, cookbook_status],
    )

    # LUT hot-reload wiring
    reload_luts_btn.click(
        fn=on_reload_luts,
        inputs=[],
        outputs=[reload_luts_status],
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

    # Name → Gradio component map.  Keyed by PROCESS_INPUT_KEYS so that adding
    # a ParamSpec (which auto-inserts a name into PROCESS_INPUT_KEYS via
    # param_names()) forces a matching component entry here.  The positional
    # `_process_inputs` list is ALWAYS DERIVED from this map, never hand-ordered,
    # so a forgotten/misplaced insertion can no longer silently shift every
    # later argument by one (the historical "brightness reads as contrast"
    # footgun).  The import-time drift guard below turns any mismatch into a
    # loud AssertionError instead of corrupted output.
    _process_input_components = {
        "img_paths": img_input,
        "recipe": recipe,
        "smooth": smooth,
        "mid_reduction": mid_reduction,
        "blotch_reduction": _blotch_reduction_state,
        "texture_opacity": texture_opacity,
        "pore_synthesis": pore_synthesis,
        "nose_smooth": nose_smooth,
        "regional_modulation": regional_modulation,
        "smooth_engine": smooth_engine,
        "undereye_shadow_strength": undereye_shadow_strength,
        "freckle_removal": freckle_removal,
        "micro_restore": micro_restore,
        "micro_dodge_burn": _micro_dodge_burn_state,
        "redness_even": _redness_even_state,
        "whiten_hue_stable": _whiten_hue_stable_state,
        "whiten": whiten,
        "equalize": equalize,
        "blemish": blemish,
        "whiten_tone": whiten_tone,
        "nose_blush": nose_blush,
        "under_eye_blush": under_eye_blush,
        "white_costume_lift": white_costume_lift,
        "dodge_burn": dodge_burn,
        "relight": relight,
        "relight_azimuth": relight_azimuth,
        "relight_elevation": relight_elevation,
        "sculpt": sculpt,
        "shine_removal": shine_removal,
        "wrinkle_soften": wrinkle_soften,
        "wrinkle_soften_forehead": _wrinkle_soften_forehead_state,
        "wrinkle_soften_nasolabial": _wrinkle_soften_nasolabial_state,
        "wrinkle_soften_neck": _wrinkle_soften_neck_state,
        "texture_transplant": texture_transplant,
        "body_smooth": body_smooth,
        "body_equalize": body_equalize,
        "body_whiten": body_whiten,
        "body_match_face": body_match_face,
        "body_relight": body_relight,
        "body_dodge_burn": body_dodge_burn,
        "shadow_lift": shadow_lift,
        "face_exposure": face_exposure,
        "body_shadow_lift": body_shadow_lift,
        "nose_restore": nose_restore,
        "skin_sss": skin_sss,
        "specular_bloom": specular_bloom,
        "specular_bloom_tone": specular_bloom_tone,
        "specular_finish": _specular_finish_state,
        "specular_finish_strength": _specular_finish_strength_state,
        "specular_recolor": _specular_recolor_state,
        "albedo_even": _albedo_even_state,
        "makeup_coverage_even": _makeup_coverage_even_state,
        "makeup_cake_reduce": _makeup_cake_reduce_state,
        "hemoglobin_smooth": _hemoglobin_smooth_state,
        "mole_protect": mole_protect,
        "vein_attenuate": _vein_attenuate_state,
        "skin_flatten": skin_flatten,
        "skin_quantize": skin_quantize,
        "skin_unify": skin_unify,
        "skin_unify_hue": skin_unify_hue,
        "skin_hue_unify": _skin_hue_unify_state,
        "skin_chroma_even": _skin_chroma_even_state,
        "skin_glow": skin_glow,
        "eye_enhance": eye_enhance,
        "catchlight": catchlight,
        "dark_circles": dark_circles,
        "undereye_darken_removal": undereye_darken_removal,
        "undereye_puffiness_reduction": undereye_puffiness_reduction,
        "eye_sclera_brighten": eye_sclera_brighten,
        "eye_sclera_vessel_remove": _eye_sclera_vessel_remove_state,
        "backdrop_cleanup": _backdrop_cleanup_state,
        "fabric_wrinkle_smooth": _fabric_wrinkle_smooth_state,
        "eye_iris_saturate": eye_iris_saturate,
        "eye_iris_hue_shift": eye_iris_hue_shift,
        "eye_iris_brightness": eye_iris_brightness,
        "teeth_whiten": teeth_whiten,
        "lip_enhance": lip_enhance,
        "lip_tint": lip_tint,
        "lip_finish": lip_finish,
        "blush": blush,
        "slimming": slimming,
        "reshape_eye_size": _reshape_eye_size_state,
        "reshape_eye_distance": _reshape_eye_distance_state,
        "reshape_nose_width": _reshape_nose_width_state,
        "reshape_nose_length": _reshape_nose_length_state,
        "reshape_jaw_width": _reshape_jaw_width_state,
        "reshape_chin_length": _reshape_chin_length_state,
        "reshape_mouth_size": _reshape_mouth_size_state,
        "reshape_smile": _reshape_smile_state,
        "reshape_forehead": _reshape_forehead_state,
        "reshape_jaw_width_l": _reshape_jaw_width_l_state,
        "reshape_jaw_width_r": _reshape_jaw_width_r_state,
        "reshape_nose_width_l": _reshape_nose_width_l_state,
        "reshape_nose_width_r": _reshape_nose_width_r_state,
        "reshape_eye_size_l": _reshape_eye_size_l_state,
        "reshape_eye_size_r": _reshape_eye_size_r_state,
        "reshape_neck_width": _reshape_neck_width_state,
        "reshape_neck_length": _reshape_neck_length_state,
        "hair_enhance": hair_enhance,
        "hair_deglare": _hair_deglare_state,
        "hair_ring_position": _hair_ring_position_state,
        "hair_ring_tint": _hair_ring_tint_state,
        "hair_remove_flyaways": _hair_remove_flyaways_state,
        "contrast": contrast,
        "brightness": brightness,
        "highlights": highlights,
        "shadows": shadows,
        "whites": whites,
        "blacks": blacks,
        "clarity": clarity,
        "vibrance": vibrance,
        "saturation": saturation,
        "auto_exposure": auto_exposure,
        "bloom": bloom,
        "bloom_threshold": bloom_threshold,
        "bloom_softness": bloom_softness,
        "glow": glow,
        "vignette": vignette,
        "sharpen": sharpen,
        "sharpen_radius": sharpen_radius,
        "subject_separation": subject_separation,
        "impact": impact,
        "fade_toe": fade_toe,
        "highlight_drift": highlight_drift,
        "airy_haze": airy_haze,
        "clarity_split_neg": clarity_split_neg,
        "clarity_split_pos": clarity_split_pos,
        "color_grade": color_grade,
        "grade_intensity": grade_intensity,
        "chromatic_aberration": chromatic_aberration,
        "grain": grain,
        "halation": halation,
        "lut": lut,
        "tonal_curve_strength": tonal_curve_strength,
        "skin_protect_strength": skin_protect_strength,
        "grain_strength": grain_strength,
        "highlight_rolloff_strength": highlight_rolloff_strength,
        "gamut_compress": _gamut_compress_state,
        "saturation_mode": _saturation_mode_state,
        "hsl_hue_red": _hsl_hue_red_state,
        "hsl_sat_red": _hsl_sat_red_state,
        "hsl_lum_red": _hsl_lum_red_state,
        "hsl_hue_orange": _hsl_hue_orange_state,
        "hsl_sat_orange": _hsl_sat_orange_state,
        "hsl_lum_orange": _hsl_lum_orange_state,
        "hsl_hue_yellow": _hsl_hue_yellow_state,
        "hsl_sat_yellow": _hsl_sat_yellow_state,
        "hsl_lum_yellow": _hsl_lum_yellow_state,
        "hsl_hue_green": _hsl_hue_green_state,
        "hsl_sat_green": _hsl_sat_green_state,
        "hsl_lum_green": _hsl_lum_green_state,
        "hsl_hue_cyan": _hsl_hue_cyan_state,
        "hsl_sat_cyan": _hsl_sat_cyan_state,
        "hsl_lum_cyan": _hsl_lum_cyan_state,
        "hsl_hue_blue": _hsl_hue_blue_state,
        "hsl_sat_blue": _hsl_sat_blue_state,
        "hsl_lum_blue": _hsl_lum_blue_state,
        "hsl_hue_purple": _hsl_hue_purple_state,
        "hsl_sat_purple": _hsl_sat_purple_state,
        "hsl_lum_purple": _hsl_lum_purple_state,
        "hsl_hue_magenta": _hsl_hue_magenta_state,
        "hsl_sat_magenta": _hsl_sat_magenta_state,
        "hsl_lum_magenta": _hsl_lum_magenta_state,
        "calibration_red_hue": _calibration_red_hue_state,
        "calibration_red_sat": _calibration_red_sat_state,
        "calibration_red_lum": _calibration_red_lum_state,
        "calibration_green_hue": _calibration_green_hue_state,
        "calibration_green_sat": _calibration_green_sat_state,
        "calibration_green_lum": _calibration_green_lum_state,
        "calibration_blue_hue": _calibration_blue_hue_state,
        "calibration_blue_sat": _calibration_blue_sat_state,
        "calibration_blue_lum": _calibration_blue_lum_state,
        "film_enable": _film_enable_state,
        "film_strength": _film_strength_state,
        "film_toe_r": _film_toe_r_state,
        "film_toe_g": _film_toe_g_state,
        "film_toe_b": _film_toe_b_state,
        "film_shoulder_r": _film_shoulder_r_state,
        "film_shoulder_g": _film_shoulder_g_state,
        "film_shoulder_b": _film_shoulder_b_state,
        "film_midpoint": _film_midpoint_state,
        "film_gamma": _film_gamma_state,
        "film_crosstalk_cy_mg": _film_crosstalk_cy_mg_state,
        "film_crosstalk_cy_ye": _film_crosstalk_cy_ye_state,
        "film_crosstalk_mg_ye": _film_crosstalk_mg_ye_state,
        "film_tonemap_strength": _film_tonemap_strength_state,
        "film_tonemap_toe": _film_tonemap_toe_state,
        "film_tonemap_shoulder": _film_tonemap_shoulder_state,
        "film_skew": _film_skew_state,
        "background_harmonize": _background_harmonize_state,
        "background_harmonize_mode": _background_harmonize_mode_state,
        "background_blur": _background_blur_state,
        "lens_blur": _lens_blur_state,
        "background_desaturation": _background_desaturation_state,
        "light_wrap": _light_wrap_state,
        "blue_shadow_grade": _blue_shadow_grade_state,
        "cyan_midtone_grade": _cyan_midtone_grade_state,
        "subject_sharpen": _subject_sharpen_state,
        "matte_black": _matte_black_state,
        "shadow_hue": shadow_hue,
        "shadow_sat": shadow_sat,
        "midtone_hue": midtone_hue,
        "midtone_sat": midtone_sat,
        "highlight_hue": highlight_hue,
        "highlight_sat": highlight_sat,
        "white_balance_kelvin": white_balance_kelvin,
        "white_balance_tint": white_balance_tint,
        "bw_channel_mixer_r": bw_channel_mixer_r,
        "bw_channel_mixer_g": bw_channel_mixer_g,
        "bw_channel_mixer_b": bw_channel_mixer_b,
        "negative_split_tone_shadow": negative_split_tone_shadow,
        "negative_split_tone_highlight": negative_split_tone_highlight,
        "hsl_hue_global": hsl_hue_global,
        "hsl_sat_global": hsl_sat_global,
        "hsl_lum_global": hsl_lum_global,
        "ai_denoise": _ai_denoise_state,
        "ai_sr_scale": _ai_sr_scale_state,
        "mv2_eyeshadow": _mv2_eyeshadow_state,
        "mv2_eyeshadow_color": _mv2_eyeshadow_color_state,
        "mv2_eyeshadow_style": _mv2_eyeshadow_style_state,
        "mv2_eyeliner": _mv2_eyeliner_state,
        "mv2_eyeliner_color": _mv2_eyeliner_color_state,
        "mv2_eyeliner_style": _mv2_eyeliner_style_state,
        "mv2_contour": _mv2_contour_state,
        "mv2_brows": _mv2_brows_state,
        "mv2_brows_color": _mv2_brows_color_state,
        "mv2_ombre": _mv2_ombre_state,
        "mv2_ombre_color1": _mv2_ombre_color1_state,
        "mv2_ombre_color2": _mv2_ombre_color2_state,
        "neural_stray_hair_boost": _neural_stray_hair_boost_state,
        "neural_defect_boost": _neural_defect_boost_state,
        "cosplay_wig_lace_blend": cosplay_wig_lace_blend,
        "cosplay_stockings_smooth": cosplay_stockings_smooth,
        "cosplay_consistency_strength": cosplay_consistency_strength,
        "body_reshape_arm_length": body_reshape_arm_length,
        "body_reshape_leg_length": body_reshape_leg_length,
        "body_reshape_torso_width": body_reshape_torso_width,
        "body_reshape_shoulder_width": body_reshape_shoulder_width,
        "body_reshape_hip_width": body_reshape_hip_width,
        "auto_body_reshape": auto_body_reshape,
        "color_ref_img": color_ref_img,
        "color_ref_strength": color_ref_strength,
        "show_compare": show_compare,
        "fast": fast,
        "export_fmt": export_fmt,
        "export_quality": export_quality,
        "export_res": export_res,
        "quality_tier": quality_tier,
        "debug_mode": debug_mode,
        "look_params": _look_params_state,
        "face_params": _face_params_state,
        "face_params_json": face_params_json,
    }
    # Drift guard: every PROCESS_INPUT_KEYS name must map to exactly one
    # component, and every component key must be a PROCESS_INPUT_KEYS name.
    # A new ParamSpec that forgets to add its component here will fail import
    # loudly instead of silently corrupting positional argument binding.
    _missing_components = set(PROCESS_INPUT_KEYS) - set(_process_input_components)
    _extra_components = set(_process_input_components) - set(PROCESS_INPUT_KEYS)
    if _missing_components or _extra_components:
        raise AssertionError(
            "_process_input_components drift vs PROCESS_INPUT_KEYS: "
            f"missing={sorted(_missing_components)} extra={sorted(_extra_components)}"
        )
    _process_inputs = [_process_input_components[k] for k in PROCESS_INPUT_KEYS]
    _process_outputs = [img_output, compare_viewer, _original_state, export_file, status, debug_gallery, debug_panel, qa_status]

    # F2: Undo/Redo history.  Push the current slider state onto the stack
    # after every mutation so undo_handler/redo_handler (wired below) can step
    # through history.  Registered after _process_inputs exists; each source's
    # mutation listener was registered earlier, so push fires post-mutation.
    recipe.change(push_undo_handler, inputs=_process_inputs, outputs=[_undo_stack_state, undo_btn, redo_btn])
    custom_style_preset.change(push_undo_handler, inputs=_process_inputs, outputs=[_undo_stack_state, undo_btn, redo_btn])
    for _reset_src in (
        reset_btn, reset_skin_smooth_btn, reset_skin_tone_btn, reset_basic_tone_btn,
        reset_tone_curve_btn, reset_relighting_btn, reset_eyes_lips_btn,
        reset_face_reshaping_btn, reset_structure_effects_btn, reset_color_grading_btn,
        reset_film_effects_btn, reset_split_toning_btn, reset_color_transfer_btn,
        reset_debug_btn, reset_body_skin_btn, reset_lch_btn, smart_process_btn,
    ):
        _reset_src.click(push_undo_handler, inputs=_process_inputs, outputs=[_undo_stack_state, undo_btn, redo_btn])

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

    # F10: Smart Process — analyze first image, auto-fill sliders + show
    # explanation readout. Outputs: recipe Radio, slider tuple (matching
    # _recipe_outputs), status Textbox, smart-analysis HTML.
    smart_process_btn.click(
        fn=on_smart_process,
        inputs=[img_input, recipe] + list(_process_inputs[1:]),
        outputs=[recipe] + list(_recipe_outputs) + [status, smart_analysis_html],
        concurrency_limit=1,
    )

    # F2: Session wiring
    save_session_btn.click(
        fn=save_session_handler,
        inputs=_process_inputs,
        outputs=[session_download],
    )

    load_session_file.change(
        fn=load_session_handler,
        inputs=[load_session_file] + _process_inputs,
        outputs=list(_process_inputs),
    )

    undo_btn.click(
        fn=undo_handler,
        inputs=[_undo_stack_state],
        outputs=_process_inputs + [_undo_stack_state, undo_btn, redo_btn],
    )

    redo_btn.click(
        fn=redo_handler,
        inputs=[_undo_stack_state],
        outputs=_process_inputs + [_undo_stack_state, undo_btn, redo_btn],
    )

    save_snapshot_btn.click(
        fn=save_snapshot_handler,
        inputs=[snapshot_name] + _process_inputs + [_snapshot_state],
        outputs=[snapshot_dropdown, _snapshot_state],
    )

    compare_snapshot_btn.click(
        fn=compare_snapshot_handler,
        inputs=[snapshot_dropdown] + _process_inputs + [_snapshot_state],
        outputs=[gr.Textbox(visible=False)],
    )

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860)
