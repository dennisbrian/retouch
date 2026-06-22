#!/usr/bin/env python3

import sys
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine
from retouch.engine import resolve_recipe
from retouch.io import imread_exif
from retouch.recipes import RECIPES
from retouch.grading import list_available_presets
from retouch.style_library import list_styles, save_style_profile, learn_dataset_style
from retouch.batch_processor import BatchProcessor
from retouch.style import StyleProfile

RECIPE_NAMES = list(RECIPES.keys())
COLOR_GRADE_NAMES = ["none"] + list_available_presets()
LUT_CHOICES = ["none", "kodak", "fuji"]
WHITEN_TONE_CHOICES = ["rosy", "porcelain", "neutral"]
LIP_FINISH_CHOICES = ["gloss", "matte", "velvet"]
SPECULAR_BLOOM_TONE_CHOICES = ["rosy", "porcelain", "neutral"]

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = RetouchEngine()
    return _engine


def recipe_defaults(recipe_name):
    rec = resolve_recipe(recipe_name)
    return {
        "smooth": int(rec["frequency"]["smooth"] * 100),
        "mid_reduction": rec["frequency"].get("mid_reduction", 0.45),
        "texture_opacity": rec["texture"].get("opacity", 1.0),
        "pore_synthesis": int(rec["texture"].get("pore_synthesis", 0.0) * 100),
        "nose_smooth": 0,
        "whiten": int(rec["skin"].get("rosy", rec["skin"].get("porcelain", 0)) * 100),
        "equalize": int(rec["skin"].get("equalize", 0) * 100),
        "relight": int(rec.get("skin", {}).get("relight", rec.get("relight_strength", 0.0) / 100.0) * 100),
        "relight_azimuth": int(rec.get("skin", {}).get("relight_azimuth", rec.get("light_azimuth", 0.0))),
        "relight_elevation": int(rec.get("skin", {}).get("relight_elevation", rec.get("light_elevation", 30.0))),
        "eye_enhance": int(rec["eyes"].get("whites", rec["eyes"].get("iris", 0)) * 100),
        "lip_enhance": int(rec["lips"].get("gloss", 0) * 100),
        "lip_tint": rec["lips"].get("tint") or "none",
        "blush": int(rec.get("blush", 0.0)),
        "teeth_whiten": int(rec["eyes"].get("whites", 0) * 100),
        "hair_enhance": int(rec["hair"].get("shine", 0) * 100),
        "dodge_burn": int(
            (rec.get("dodge_burn", {}).get("amount", 0.0) if isinstance(rec.get("dodge_burn"), dict)
             else rec.get("dodge_burn", 0.0) / 100.0) * 100
        ),
        "specular_bloom": rec.get("specular_bloom", 0),
        "bloom": int(rec.get("bloom", {}).get("opacity", 0.0) * 100),
        "bloom_threshold": int(rec.get("bloom", {}).get("threshold", 210.0)),
        "bloom_softness": int(rec.get("bloom", {}).get("softness", 30.0)),
        "contrast": rec.get("contrast", 0),
        "brightness": int(rec.get("brightness", 0.0)),
        "highlights": int(rec.get("highlights", 0.0)),
        "shadows": int(rec.get("shadows", 0.0)),
        "whites": int(rec.get("whites", 0.0)),
        "blacks": int(rec.get("blacks", 0.0)),
        "nose_blush": bool(rec.get("nose_blush", False)),
        "under_eye_blush": bool(rec.get("under_eye_blush", False)),
        "white_costume_lift": bool(rec.get("white_costume_lift", False)),
        # --- CLI-parity params ---
        "blemish": int(rec.get("frequency", {}).get("smooth", 0.30) * 100),
        "dark_circles": int(rec.get("eyes", {}).get("dark_circles", 0.0) * 100),
        "whiten_tone": ("porcelain" if "porcelain" in rec.get("skin", {}) else "rosy"),
        "auto_exposure": False,
        "lip_finish": rec.get("lip_finish", "gloss"),
        "slimming": int(rec.get("slimming", 0.0)),
        "impact": int(rec.get("finish", {}).get("impact", 0.0) * 100),
        # --- Engine advanced params ---
        "clarity": int(rec.get("clarity", 0.0)),
        "vibrance": int(rec.get("vibrance", 0.0)),
        "saturation": int(rec.get("saturation", 0.0)),
        "glow": int(rec.get("glow", 0.0)),
        "vignette": int(rec.get("vignette", 0.0)),
        "sharpen": int(rec.get("sharpen", 0.0)),
        "sharpen_radius": float(rec.get("sharpen_radius", 1.0)),
        "subject_separation": int(rec.get("subject_separation", 0.0) if rec.get("subject_separation", 0.0) > 1.0 else rec.get("subject_separation", 0.0) * 100),
        "specular_bloom_tone": rec.get("specular_bloom_tone", "rosy"),
        "color_grade": rec.get("color_harmony", {}).get("preset", "none"),
        "grade_intensity": int(rec.get("color_harmony", {}).get("amount", 0.0) * 100),
        "chromatic_aberration": float(rec.get("chromatic_aberration", 0.0)),
        "grain": float(rec.get("grain", 0.0)),
        "halation": float(rec.get("halation", 0.0)),
        "lut": rec.get("lut", "none"),
        # --- Split toning ---
        "shadow_hue": int(rec.get("shadow_hue", 0.0)),
        "shadow_sat": int(rec.get("shadow_sat", 0.0)),
        "midtone_hue": int(rec.get("midtone_hue", 0.0)),
        "midtone_sat": int(rec.get("midtone_sat", 0.0)),
        "highlight_hue": int(rec.get("highlight_hue", 0.0)),
        "highlight_sat": int(rec.get("highlight_sat", 0.0)),
    }


def get_custom_style_names():
    styles = list_styles()
    return [s["name"] for s in styles]


def apply_custom_style(style_name):
    if not style_name:
        return [gr.update()]*58
    
    styles = list_styles()
    target = None
    for s in styles:
        if s["name"] == style_name:
            target = s
            break
            
    if not target:
        return [gr.update()]*58
        
    p_dict = target["profile"]
    profile = StyleProfile(**p_dict)
    
    smooth_val = int(np.clip(profile.skin_smooth_strength * 100.0, 0.0, 100.0))
    whiten_val = int(np.clip(profile.skin_l_mean_delta * 4.0, -100.0, 100.0))
    mid_red_val = float(np.clip(profile.skin_mid_reduction, 0.0, 1.0))
    tex_op_val = float(np.clip(profile.skin_texture_opacity, 0.0, 1.0))
    contrast_val = int(profile.contrast_delta)
    brightness_val = int(np.clip(profile.brightness_delta * 2.0, -100.0, 100.0))
    
    return (
        smooth_val, mid_red_val, tex_op_val, 0, 0,
        whiten_val, 0, False,
        0, 0, 30,
        5, 5,
        5, "none", 0, False, False,
        5, 0, 0, 0, 210, 30, contrast_val, brightness_val,
        0, 0, 0, 0,
        # New params (28 defaults)
        0, 0, "rosy", False,
        0, 0, 0, "gloss",
        0, 0,
        0, 1.0, 0, 0, 0, "rosy",
        "none", 0,
        0, 0, 0, "none",
        0, 0, 0, 0, 0, 0,
    )


def on_save_style(style_name, author, tags_str,
                  smooth, mid_reduction, texture_opacity, pore_synthesis,
                  whiten, contrast, brightness):
    if not style_name.strip():
        return gr.update(), gr.update(), "Error: Style name cannot be empty."

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
        return gr.update(choices=choices, value=style_name), gr.update(choices=choices, value=style_name), f"Style '{style_name}' saved successfully!"
    except Exception as e:
        return gr.update(), gr.update(), f"Failed to save style: {e}"


def on_learn_style(orig_dir, edit_dir, style_name, author, tags_str, prg=gr.Progress()):
    if not orig_dir or not edit_dir:
        return gr.update(), gr.update(), "Error: Original and Edited folders must be specified."
    if not style_name.strip():
        return gr.update(), gr.update(), "Error: Style name cannot be empty."

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
            author=author or None,
            tags=tags,
        )
        
        choices = get_custom_style_names()
        return gr.update(choices=choices, value=style_name), gr.update(choices=choices, value=style_name), f"Extracted & saved style '{style_name}' from {count} image pairs!"
    except Exception as e:
        return gr.update(), gr.update(), f"Error during dataset learning: {e}"


def on_process_folder(input_dir, output_dir, style_type, custom_style_name, recipe_name,
                      export_fmt, export_quality, export_res, auto_group, generate_sheet, export_zip,
                      prg=gr.Progress()):
    if not input_dir or not output_dir:
        return None, None, "Error: Both Input and Output directories must be specified."

    processor = BatchProcessor()
    
    profile = None
    style_val = recipe_name
    
    if style_type == "Use Custom Style":
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
        
        return sheet_path, zip_path, log
    except Exception as e:
        return None, None, f"Exception during batch processing: {e}"


def process_image(img_paths, recipe,
                  smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth,
                  whiten, equalize, white_costume_lift,
                  relight, relight_azimuth, relight_elevation,
                  eye_enhance, teeth_whiten,
                  lip_enhance, lip_tint, blush, nose_blush, under_eye_blush,
                  hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
                  highlights, shadows, whites, blacks,
                  color_ref_path, color_ref_strength,
                  show_compare, fast,
                  export_fmt, export_quality, export_res,
                  # New params
                  blemish, dark_circles, whiten_tone, auto_exposure,
                  clarity, vibrance, saturation, lip_finish,
                  slimming, impact,
                  sharpen, sharpen_radius, glow, vignette, subject_separation, specular_bloom_tone,
                  color_grade, grade_intensity,
                  chromatic_aberration, grain, halation, lut,
                  shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat,
                  debug_mode):
    if img_paths is None:
        return None, None, "Please upload an image first.", None, gr.update(visible=False)

    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    if len(img_paths) == 0:
        return None, None, "Please upload at least one image.", None, gr.update(visible=False)

    exported_paths = []
    first_result_rgb = None
    first_combined = None
    debug_images = []

    color_ref_bgr = None
    if color_ref_path is not None:
        if isinstance(color_ref_path, dict):
            color_ref_path = color_ref_path.get("name") or color_ref_path.get("path")
        elif isinstance(color_ref_path, list) and len(color_ref_path) > 0:
            ref_item = color_ref_path[0]
            if isinstance(ref_item, dict):
                color_ref_path = ref_item.get("name") or ref_item.get("path")
            else:
                color_ref_path = ref_item
        color_ref_bgr = imread_exif(color_ref_path)

    lip_tint_val = lip_tint if lip_tint != "none" else None
    color_grade_val = color_grade if color_grade != "none" else ""
    lut_val = lut if lut != "none" else None
    grade_intensity_val = grade_intensity / 100.0
    engine = get_engine()

    temp_dir = tempfile.mkdtemp()
    debug_dir = os.path.join(temp_dir, "debug") if debug_mode else None

    for idx, path_item in enumerate(img_paths):
        try:
            curr_path = path_item
            if isinstance(path_item, dict):
                curr_path = path_item.get("name") or path_item.get("path")
            
            img_bgr = imread_exif(curr_path)
            original = img_bgr.copy()

            result = engine.process(
                img_bgr,
                recipe=recipe,
                smooth=smooth,
                mid_reduction=mid_reduction,
                texture_opacity=texture_opacity,
                pore_synthesis=pore_synthesis,
                nose_smooth=nose_smooth if nose_smooth > 0 else None,
                whiten=whiten,
                whiten_tone=whiten_tone,
                equalize=equalize,
                blemish=blemish,
                white_costume_lift=white_costume_lift,
                relight=relight,
                relight_azimuth=relight_azimuth,
                relight_elevation=relight_elevation,
                eye_enhance=eye_enhance,
                dark_circles=dark_circles,
                teeth_whiten=teeth_whiten,
                lip_enhance=lip_enhance,
                lip_tint=lip_tint_val,
                lip_finish=lip_finish,
                blush=blush,
                nose_blush=nose_blush,
                under_eye_blush=under_eye_blush,
                hair_enhance=hair_enhance,
                dodge_burn=dodge_burn,
                slimming=slimming,
                impact=impact,
                specular_bloom=specular_bloom,
                specular_bloom_tone=specular_bloom_tone,
                bloom=bloom,
                bloom_threshold=bloom_threshold,
                bloom_softness=bloom_softness,
                contrast=contrast,
                brightness=brightness,
                highlights=highlights,
                shadows=shadows,
                whites=whites,
                blacks=blacks,
                clarity=clarity,
                vibrance=vibrance,
                saturation=saturation,
                glow=glow,
                vignette=vignette,
                sharpen=sharpen,
                sharpen_radius=sharpen_radius,
                subject_separation=subject_separation,
                color_grade=color_grade_val,
                grade_intensity=grade_intensity_val,
                chromatic_aberration=chromatic_aberration if chromatic_aberration > 0 else None,
                grain=grain if grain > 0 else None,
                halation=halation if halation > 0 else None,
                lut=lut_val,
                shadow_hue=shadow_hue,
                shadow_sat=shadow_sat,
                midtone_hue=midtone_hue,
                midtone_sat=midtone_sat,
                highlight_hue=highlight_hue,
                highlight_sat=highlight_sat,
                auto_exposure=auto_exposure,
                color_ref=color_ref_bgr,
                color_transfer_intensity=color_ref_strength,
                fast=fast,
                debug_dir=debug_dir if idx == 0 else None,
            )

            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            if idx == 0:
                first_result_rgb = result_rgb
                if show_compare:
                    h = min(original.shape[0], result.shape[0])
                    sep = np.full((h, 4, 3), 200, dtype=np.uint8)
                    orig_rgb = cv2.cvtColor(original[:h], cv2.COLOR_BGR2RGB)
                    res_rgb = result_rgb[:h]
                    combined = np.hstack([orig_rgb, sep, res_rgb])
                    max_h = 900
                    if combined.shape[0] > max_h:
                        scale = max_h / combined.shape[0]
                        new_w = int(combined.shape[1] * scale)
                        combined = cv2.resize(combined, (new_w, max_h), interpolation=cv2.INTER_AREA)
                    first_combined = combined
                else:
                    if first_result_rgb.shape[0] > 900:
                        scale = 900 / first_result_rgb.shape[0]
                        new_w = int(first_result_rgb.shape[1] * scale)
                        first_result_rgb = cv2.resize(first_result_rgb, (new_w, 900), interpolation=cv2.INTER_AREA)

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
            export_max = {"Original": None, "4K (3840px)": 3840, "2K (2048px)": 2048,
                          "Full HD (1920px)": 1920, "HD (1280px)": 1280, "720px": 720}.get(export_res)
            if export_max is not None:
                h, w = export_img.shape[:2]
                if max(h, w) > export_max:
                    scale = export_max / max(h, w)
                    export_img = cv2.resize(export_img, (int(w * scale), int(h * scale)),
                                            interpolation=cv2.INTER_AREA)

            ext = {"JPEG": ".jpg", "PNG": ".png", "WebP": ".webp"}.get(export_fmt, ".jpg")
            filename = Path(curr_path).stem
            out_path = os.path.join(temp_dir, f"{filename}_retouched{ext}")
            write_params = []
            if export_fmt == "JPEG":
                write_params = [cv2.IMWRITE_JPEG_QUALITY, export_quality]
            elif export_fmt == "WebP":
                write_params = [cv2.IMWRITE_WEBP_QUALITY, export_quality]
            cv2.imwrite(out_path, export_img, write_params)
            exported_paths.append(out_path)

        except Exception as e:
            print(f"Failed to process {path_item}: {e}")
            from retouch.utils import log_crash
            crash_path = log_crash(e, {
                "recipe": recipe,
                "image_path": str(curr_path),
                "show_compare": show_compare,
                "fast": fast
            })
            if crash_path:
                print(f"Crash details saved to: {crash_path}")

    if not exported_paths:
        return None, None, "Error: No images were successfully processed.", None, gr.update(visible=False)

    preview = first_combined if show_compare else first_result_rgb
    debug_gallery = debug_images if debug_images else None
    debug_vis = gr.update(visible=bool(debug_images))

    if len(exported_paths) > 1:
        zip_path = os.path.join(tempfile.gettempdir(), "retouch_batch_export.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for exp_path in exported_paths:
                zipf.write(exp_path, arcname=os.path.basename(exp_path))
        return preview, zip_path, f"Processed {len(exported_paths)}/{len(img_paths)} images successfully ✓", debug_gallery, debug_vis
    else:
        return preview, exported_paths[0], "Done ✓", debug_gallery, debug_vis


def on_recipe_change(recipe):
    d = recipe_defaults(recipe)
    return (
        d["smooth"], d["mid_reduction"], d["texture_opacity"], d["pore_synthesis"], d["nose_smooth"],
        d["whiten"], d["equalize"], d["white_costume_lift"],
        d["relight"], d["relight_azimuth"], d["relight_elevation"],
        d["eye_enhance"], d["teeth_whiten"],
        d["lip_enhance"], d["lip_tint"], d["blush"], d["nose_blush"], d["under_eye_blush"],
        d["hair_enhance"], d["dodge_burn"], d["specular_bloom"], d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["contrast"], d["brightness"],
        d["highlights"], d["shadows"], d["whites"], d["blacks"],
        # New params (28)
        d["blemish"], d["dark_circles"], d["whiten_tone"], d["auto_exposure"],
        d["clarity"], d["vibrance"], d["saturation"], d["lip_finish"],
        d["slimming"], d["impact"],
        d["sharpen"], d["sharpen_radius"], d["glow"], d["vignette"], d["subject_separation"], d["specular_bloom_tone"],
        d["color_grade"], d["grade_intensity"],
        d["chromatic_aberration"], d["grain"], d["halation"], d["lut"],
        d["shadow_hue"], d["shadow_sat"], d["midtone_hue"], d["midtone_sat"], d["highlight_hue"], d["highlight_sat"],
    )


LIP_TINTS = ["none", "cosplay", "rose", "pink", "coral", "natural", "berry"]
custom_style_choices = get_custom_style_names()

with gr.Blocks(title="🪄 Retouch — AI Portrait Workflow Platform", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan"), css="""
    /* Premium font family import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    body, input, button, select, textarea {
        font-family: 'Outfit', -apple-system, sans-serif !important;
    }
    
    /* Elegant background gradient */
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    .dark .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%) !important;
    }
    
    /* Navigation Tabs Container */
    .tabs {
        border-bottom: none !important;
        background: rgba(255, 255, 255, 0.6) !important;
        border-radius: 16px !important;
        padding: 6px !important;
        box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.04), 0 4px 20px -2px rgba(0,0,0,0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        margin-bottom: 20px !important;
    }
    .dark .tabs {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.2) !important;
    }
    
    .tab-nav {
        border-bottom: none !important;
        display: flex !important;
        gap: 6px !important;
    }
    
    .tab-nav button {
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        background: transparent !important;
        transition: all 0.2s ease-in-out !important;
    }
    .dark .tab-nav button {
        color: #94a3b8 !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%) !important;
        color: white !important;
        box-shadow: 0 4px 14px -2px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Custom buttons with nice gradient and shadows */
    .primary-btn {
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-radius: 12px !important;
        cursor: pointer !important;
    }
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px 0 rgba(99, 102, 241, 0.5) !important;
    }
    .primary-btn:active {
        transform: translateY(0px) !important;
    }
    
    /* Secondary Action Button Styling */
    .secondary-btn {
        border: 1px solid #cbd5e1 !important;
        background: white !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    .dark .secondary-btn {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: #1e293b !important;
        color: #f1f5f9 !important;
    }
    .secondary-btn:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        transform: translateY(-1px) !important;
    }
    .dark .secondary-btn:hover {
        background: #334155 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Group panel/card styling */
    .gr-group {
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 4px 18px -2px rgba(0,0,0,0.03) !important;
        margin-bottom: 15px !important;
    }
    .dark .gr-group {
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        background: rgba(30, 41, 59, 0.5) !important;
        box-shadow: 0 4px 18px -2px rgba(0,0,0,0.2) !important;
    }
    
    /* Accordion styles */
    .accordion {
        border: 1px solid #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .dark .accordion {
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: #1e293b !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    .accordion:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.04) !important;
        border-color: #cbd5e1 !important;
    }
    .dark .accordion:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.3) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Slider visual tuning */
    .gr-slider input[type=range] {
        accent-color: #6366f1 !important;
    }
""") as app:
    # Beautiful Glassmorphism Header
    with gr.Group():
        gr.HTML("""
        <div style="text-align: center; padding: 0.5rem 0; font-family: 'Outfit', sans-serif;">
            <h1 style="margin: 0; font-size: 2.35rem; font-weight: 800; background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;">🪄 Retouch — AI Portrait Workflow Platform</h1>
            <p style="font-size: 1.05rem; color: #64748b; margin-top: 0.4rem; font-weight: 500;">Professional high-fidelity skin retouching, virtual studio relighting, and custom style workflows</p>
        </div>
        """)

    with gr.Tabs():
        with gr.Tab("📸 Single & Multi Photo Retouching"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.File(label="Input Image(s) (RAW supported)", file_types=["image"], file_count="multiple")

                    with gr.Group():
                        with gr.Row():
                            recipe = gr.Dropdown(
                                choices=RECIPE_NAMES, value="natural", label="Base Preset Recipe",
                                info="Select a preset recipe to auto-fill sliders"
                            )
                            custom_style_preset = gr.Dropdown(
                                choices=custom_style_choices, value=None, label="Or Load Custom Style Profile", interactive=True,
                                info="Select an extracted style from your library"
                            )

                        show_compare = gr.Checkbox(label="Show side-by-side comparison screen", value=True)

                        with gr.Row():
                            fast = gr.Checkbox(label="Fast Preview (Recommended)", value=True,
                                               info="Downsamples image to speed up interactive updates")
                        
                        with gr.Row():
                            process_btn = gr.Button("Process Image(s) ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])
                            reset_btn = gr.Button("Reset Overrides 🔄", variant="secondary", size="lg", elem_classes=["secondary-btn"])

                    with gr.Group():
                        gr.Markdown("### ⚙️ Manual Overrides")
                        
                        with gr.Accordion("✨ Skin Smoothing & Texture", open=True):
                            smooth = gr.Slider(0, 100, 30, step=1, label="Smooth", info="Strength of skin smoothing (blur/median blend)")
                            nose_smooth = gr.Slider(0, 100, 0, step=1, label="Nose Smooth (0 = follow face)", info="Additional smoothing for nose bridge highlights")
                            mid_reduction = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Mid Frequency Reduction", info="Target mid-level skin blemishes while preserving high-frequency pores")
                            texture_opacity = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Texture Opacity", info="Control original pore structure opacity overlay")
                            pore_synthesis = gr.Slider(0, 100, 0, step=1, label="Pore Synthesis", info="Add micro-texture/synthesized pores to prevent artificial plastic skin")
                            blemish = gr.Slider(0, 100, 0, step=1, label="Blemish Removal", info="AI blemish detection and inpainting for acne/spots")

                        with gr.Accordion("🎨 Skin & Tone", open=False):
                            whiten = gr.Slider(0, 100, 10, step=1, label="Whitening", info="Luminance boost and porcelain skin color match")
                            whiten_tone = gr.Dropdown(choices=WHITEN_TONE_CHOICES, value="rosy", label="Whitening Tone", interactive=True, info="Tone direction: rosy (warm pink), porcelain (cool neutral), neutral")
                            equalize = gr.Slider(0, 100, 20, step=1, label="Equalize", info="Even out skin redness and regional color inconsistencies")
                            auto_exposure = gr.Checkbox(label="Auto Exposure Correction", value=False, info="Automatically correct under/over-exposed images before processing")
                            white_costume_lift = gr.Checkbox(label="White Costume Lift", value=False, info="Selectively boost bright clothing to create separation")
                            contrast = gr.Slider(-50, 50, 0, step=1, label="Contrast", info="Adjust global image contrast")
                            brightness = gr.Slider(-50, 50, 0, step=1, label="Brightness", info="Adjust global image brightness")
                            clarity = gr.Slider(-100, 100, 0, step=1, label="Clarity", info="Mid-tone contrast / local contrast enhancement (negative = soften)")
                            vibrance = gr.Slider(-100, 100, 0, step=1, label="Vibrance", info="Smart saturation boost that protects skin tones")
                            saturation = gr.Slider(-100, 100, 0, step=1, label="Saturation", info="Uniform global saturation adjustment")
                            gr.Markdown("**Tone Curve Controls**")
                            highlights = gr.Slider(-100, 100, 0, step=1, label="Highlights", info="Recover or boost bright highlight regions")
                            shadows = gr.Slider(-100, 100, 0, step=1, label="Shadows", info="Open up or deepen shadow regions")
                            whites = gr.Slider(-100, 100, 0, step=1, label="Whites", info="Control absolute white point ceiling")
                            blacks = gr.Slider(-100, 100, 0, step=1, label="Blacks", info="Control absolute black point floor")

                        with gr.Accordion("💡 Virtual Studio Relighting", open=False):
                            relight = gr.Slider(0, 100, 0, step=1, label="Relight Strength", info="Intensity of 3D virtual studio light source redirection")
                            relight_azimuth = gr.Slider(-180, 180, 0, step=1, label="Light Azimuth", info="Horizontal light source direction angle (-180° to 180°)")
                            relight_elevation = gr.Slider(-90, 90, 30, step=1, label="Light Elevation", info="Vertical light source direction angle (-90° to 90°)")

                        with gr.Accordion("👁️ Eyes & Lips", open=False):
                            eye_enhance = gr.Slider(0, 100, 5, step=1, label="Eye Enhance", info="Boost eye clarity, iris reflection details, and whites brightness")
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
                            slimming = gr.Slider(0, 100, 0, step=1, label="Face Slimming", info="Liquify-based face slimming/reshaping via landmark-driven warp")

                        with gr.Accordion("🌟 Structure & Effects", open=False):
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
                            color_grade = gr.Dropdown(choices=COLOR_GRADE_NAMES, value="none", label="Color Grade Preset", interactive=True, info="Apply a film/color grading preset from the presets library")
                            grade_intensity = gr.Slider(0, 100, 100, step=1, label="Grade Intensity", info="Blend strength of the color grade (0-100%)")

                        with gr.Accordion("🎞️ Film & Analog Effects", open=False):
                            chromatic_aberration = gr.Slider(0, 20, 0, step=0.5, label="Chromatic Aberration", info="Lens fringing effect (RGB channel shift in pixels)")
                            grain = gr.Slider(0, 100, 0, step=1, label="Film Grain", info="Analog film grain noise overlay")
                            halation = gr.Slider(0, 100, 0, step=1, label="Halation", info="Red light bloom around bright highlights (analog film artifact)")
                            lut = gr.Dropdown(choices=LUT_CHOICES, value="none", label="Film Emulation LUT", interactive=True, info="Apply a film stock emulation LUT (Kodak / Fuji)")

                        with gr.Accordion("🌈 Split Toning", open=False):
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
                            gr.Markdown("Upload a reference image to match its color tone using CDF-based histogram transfer")
                            color_ref_img = gr.File(label="Reference Image (RAW supported)", file_types=["image"])
                            color_ref_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Transfer Strength", info="Mix ratio between original grade and matched reference grade")

                        with gr.Accordion("🔍 Debug & Mask Preview", open=False):
                            debug_mode = gr.Checkbox(label="Generate Debug Masks", value=False, info="Save skin/lips/frequency-layer masks and display them for tuning")

                        process_btn_bottom = gr.Button("Apply Overrides & Process ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🖼️ Output Preview")
                        img_output = gr.Image(label="Processed Result", height=540, show_label=False)
                        status = gr.Textbox(label="Status", interactive=False, placeholder="Upload an image and click Process to start...")
                        export_file = gr.File(label="📥 Download Exported Assets")

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

                    with gr.Group(visible=False) as debug_panel:
                        gr.Markdown("### 🔍 Debug Masks & Frequency Layers")
                        debug_gallery = gr.Gallery(label="Masks (skin, skin+hair, lips, sharpen, glow, freq_low, freq_mid, freq_high)", columns=4, height=300)

        with gr.Tab("📁 Folder Automation & Ingestion"):
            with gr.Row():
                with gr.Column(scale=1):
                    folder_in = gr.Textbox(label="Input Folder Path", placeholder="/path/to/photos", info="Absolute path to directory containing raw/jpeg source photos.")
                    folder_out = gr.Textbox(label="Output Folder Path", placeholder="/path/to/exports", info="Absolute path where processed results will be written.")
                    
                    with gr.Group():
                        gr.Markdown("### Style Mode")
                        batch_style_type = gr.Radio(choices=["Use Standard Recipe", "Use Custom Style"], value="Use Standard Recipe", label="Style Mode", info="Choose whether to apply a built-in recipe preset or a custom learned style profile.")
                        batch_recipe = gr.Dropdown(choices=RECIPE_NAMES, value="natural", label="Standard Recipe", info="Select standard built-in recipe preset.")
                        batch_custom_style = gr.Dropdown(choices=custom_style_choices, value=None, label="Custom Style Profile", interactive=True, info="Select a custom style profile from your library.")
                    
                    with gr.Group():
                        gr.Markdown("### Export Formatting")
                        batch_fmt = gr.Radio(choices=["JPEG", "PNG", "WebP"], value="JPEG", label="Format")
                        batch_quality = gr.Slider(10, 100, 95, step=1, label="Quality")
                        batch_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Export Resolution"
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

        with gr.Tab("🎨 Style Library & Learning"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Save Current Settings as Custom Style")
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
        smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth,
        whiten, equalize, white_costume_lift,
        relight, relight_azimuth, relight_elevation,
        eye_enhance, teeth_whiten,
        lip_enhance, lip_tint, blush, nose_blush, under_eye_blush,
        hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
        highlights, shadows, whites, blacks,
        blemish, dark_circles, whiten_tone, auto_exposure,
        clarity, vibrance, saturation, lip_finish,
        slimming, impact,
        sharpen, sharpen_radius, glow, vignette, subject_separation, specular_bloom_tone,
        color_grade, grade_intensity,
        chromatic_aberration, grain, halation, lut,
        shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat,
    ]

    recipe.change(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=_recipe_outputs,
    )

    custom_style_preset.change(
        fn=apply_custom_style,
        inputs=[custom_style_preset],
        outputs=_recipe_outputs,
    )

    reset_btn.click(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=_recipe_outputs,
    )

    save_style_btn.click(
        fn=on_save_style,
        inputs=[save_name, save_author, save_tags,
                smooth, mid_reduction, texture_opacity, pore_synthesis,
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
        smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth,
        whiten, equalize, white_costume_lift,
        relight, relight_azimuth, relight_elevation,
        eye_enhance, teeth_whiten,
        lip_enhance, lip_tint, blush, nose_blush, under_eye_blush,
        hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
        highlights, shadows, whites, blacks,
        color_ref_img, color_ref_strength,
        show_compare, fast,
        export_fmt, export_quality, export_res,
        blemish, dark_circles, whiten_tone, auto_exposure,
        clarity, vibrance, saturation, lip_finish,
        slimming, impact,
        sharpen, sharpen_radius, glow, vignette, subject_separation, specular_bloom_tone,
        color_grade, grade_intensity,
        chromatic_aberration, grain, halation, lut,
        shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat,
        debug_mode,
    ]
    _process_outputs = [img_output, export_file, status, debug_gallery, debug_panel]

    process_btn.click(
        fn=process_image,
        inputs=_process_inputs,
        outputs=_process_outputs,
    )

    process_btn_bottom.click(
        fn=process_image,
        inputs=_process_inputs,
        outputs=_process_outputs,
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
