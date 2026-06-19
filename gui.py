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
from retouch.io import imread_exif
from retouch.recipes import RECIPES
from retouch.style_library import list_styles, save_style_profile, learn_dataset_style
from retouch.batch_processor import BatchProcessor
from retouch.style import StyleProfile

RECIPE_NAMES = list(RECIPES.keys())

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = RetouchEngine()
    return _engine


def recipe_defaults(recipe_name):
    rec = RECIPES.get(recipe_name, RECIPES["natural"])
    return {
        "smooth": int(rec["frequency"]["smooth"] * 100),
        "mid_reduction": rec["frequency"].get("mid_reduction", 0.45),
        "texture_opacity": rec["texture"].get("opacity", 1.0),
        "pore_synthesis": int(rec["texture"].get("pore_synthesis", 0.0) * 100),
        "nose_smooth": 0,
        "whiten": int(rec["skin"].get("rosy", rec["skin"].get("porcelain", 0)) * 100),
        "equalize": int(rec["skin"].get("equalize", 0) * 100),
        "relight": int(rec["skin"].get("relight", 0.0) * 100),
        "relight_azimuth": int(rec["skin"].get("relight_azimuth", 0.0)),
        "relight_elevation": int(rec["skin"].get("relight_elevation", 30.0)),
        "eye_enhance": int(rec["eyes"].get("whites", rec["eyes"].get("iris", 0)) * 100),
        "lip_enhance": int(rec["lips"].get("gloss", 0) * 100),
        "lip_tint": rec["lips"].get("tint", "none"),
        "blush": 25 if recipe_name in ("cosplay", "scifi_cosplay", "cyber_doll", "anime_cosplay", "anime", "xiaohongshu", "idol", "wedding") else 0,
        "teeth_whiten": int(rec["eyes"].get("whites", 0) * 100),
        "hair_enhance": int(rec["hair"].get("shine", 0) * 100),
        "dodge_burn": int(rec["dodge_burn"].get("amount", 0) * 100),
        "specular_bloom": rec.get("specular_bloom", 0),
        "bloom": int(rec.get("bloom", {}).get("opacity", 0.0) * 100),
        "bloom_threshold": int(rec.get("bloom", {}).get("threshold", 210.0)),
        "bloom_softness": int(rec.get("bloom", {}).get("softness", 30.0)),
        "contrast": rec.get("contrast", 0),
        "brightness": 0,
        "highlights": 0,
        "shadows": 0,
        "whites": 0,
        "blacks": 0,
    }


def get_custom_style_names():
    styles = list_styles()
    return [s["name"] for s in styles]


def apply_custom_style(style_name):
    if not style_name:
        return [gr.update()]*27
    
    styles = list_styles()
    target = None
    for s in styles:
        if s["name"] == style_name:
            target = s
            break
            
    if not target:
        return [gr.update()]*27
        
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
        whiten_val, 0,
        0, 0, 30,
        5, 5,
        5, "none", 0,
        5, 0, 0, 0, 210, 30, contrast_val, brightness_val,
        0, 0, 0, 0
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
                  whiten, equalize,
                  relight, relight_azimuth, relight_elevation,
                  eye_enhance, teeth_whiten,
                  lip_enhance, lip_tint, blush,
                  hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
                  highlights, shadows, whites, blacks,
                  color_ref_path, color_ref_strength,
                  show_compare, fast,
                  export_fmt, export_quality, export_res):
    if img_paths is None:
        return None, None, "Please upload an image first."

    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    if len(img_paths) == 0:
        return None, None, "Please upload at least one image."

    exported_paths = []
    first_result_rgb = None
    first_combined = None

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
    engine = get_engine()

    temp_dir = tempfile.mkdtemp()

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
                equalize=equalize,
                relight=relight,
                relight_azimuth=relight_azimuth,
                relight_elevation=relight_elevation,
                eye_enhance=eye_enhance,
                teeth_whiten=teeth_whiten,
                lip_enhance=lip_enhance,
                lip_tint=lip_tint_val,
                blush=blush,
                hair_enhance=hair_enhance,
                dodge_burn=dodge_burn,
                specular_bloom=specular_bloom,
                bloom=bloom,
                bloom_threshold=bloom_threshold,
                bloom_softness=bloom_softness,
                contrast=contrast,
                brightness=brightness,
                highlights=highlights,
                shadows=shadows,
                whites=whites,
                blacks=blacks,
                color_ref=color_ref_bgr,
                color_transfer_intensity=color_ref_strength,
                fast=fast,
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

    if not exported_paths:
        return None, None, "Error: No images were successfully processed."

    preview = first_combined if show_compare else first_result_rgb

    if len(exported_paths) > 1:
        zip_path = os.path.join(tempfile.gettempdir(), "retouch_batch_export.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for exp_path in exported_paths:
                zipf.write(exp_path, arcname=os.path.basename(exp_path))
        return preview, zip_path, f"Processed {len(exported_paths)}/{len(img_paths)} images successfully ✓"
    else:
        return preview, exported_paths[0], "Done ✓"


def on_recipe_change(recipe):
    d = recipe_defaults(recipe)
    return (
        d["smooth"], d["mid_reduction"], d["texture_opacity"], d["pore_synthesis"], d["nose_smooth"],
        d["whiten"], d["equalize"],
        d["relight"], d["relight_azimuth"], d["relight_elevation"],
        d["eye_enhance"], d["teeth_whiten"],
        d["lip_enhance"], d["lip_tint"], d["blush"],
        d["hair_enhance"], d["dodge_burn"], d["specular_bloom"], d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["contrast"], d["brightness"],
        d["highlights"], d["shadows"], d["whites"], d["blacks"],
    )


LIP_TINTS = ["none", "cosplay", "rose", "pink", "coral", "natural", "berry"]
custom_style_choices = get_custom_style_names()

with gr.Blocks(title="Retouch GUI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Retouch — AI Portrait Workflow Platform")

    with gr.Tabs():
        with gr.Tab("Single & Multi Photo Retouching"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.File(label="Input Image(s) (RAW supported)", file_types=["image"], file_count="multiple")

                    with gr.Group():
                        recipe = gr.Dropdown(
                            choices=RECIPE_NAMES, value="natural", label="Recipe",
                            info="Select a recipe to auto-fill sliders"
                        )
                        custom_style_preset = gr.Dropdown(
                            choices=custom_style_choices, label="Or Load Custom Style",
                            info="Select a custom style from your library"
                        )

                        show_compare = gr.Checkbox(label="Show before/after comparison", value=True)

                        with gr.Row():
                            fast = gr.Checkbox(label="Fast Preview", value=True,
                                               info="Lower resolution for quicker results")
                            process_btn = gr.Button("Process", variant="primary", size="lg")

                    with gr.Group():
                        gr.Markdown("### Manual Overrides")
                        smooth = gr.Slider(0, 100, 30, step=1, label="Smooth")
                        nose_smooth = gr.Slider(0, 100, 0, step=1, label="Nose Smooth (0 = follow face)")
                        mid_reduction = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Mid Reduction")
                        texture_opacity = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Texture Opacity")
                        pore_synthesis = gr.Slider(0, 100, 0, step=1, label="Pore Synthesis")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Output", height=360)
                    export_file = gr.File(label="Download exported image")
                    with gr.Row():
                        export_fmt = gr.Radio(choices=["JPEG", "PNG", "WebP"], value="JPEG", label="Format", interactive=True)
                        export_quality = gr.Slider(10, 100, 95, step=1, label="Quality")
                    export_res = gr.Dropdown(
                        choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                        value="Original", label="Export Resolution", interactive=True)
                    status = gr.Textbox(label="Status", interactive=False)

                    with gr.Accordion("Skin & Tone", open=False):
                        whiten = gr.Slider(0, 100, 10, step=1, label="Whitening")
                        equalize = gr.Slider(0, 100, 20, step=1, label="Equalize")
                        contrast = gr.Slider(-50, 50, 0, step=1, label="Contrast")
                        brightness = gr.Slider(-50, 50, 0, step=1, label="Brightness")
                        gr.Markdown("**Tone Curve**")
                        highlights = gr.Slider(-100, 100, 0, step=1, label="Highlights")
                        shadows = gr.Slider(-100, 100, 0, step=1, label="Shadows")
                        whites = gr.Slider(-100, 100, 0, step=1, label="Whites")
                        blacks = gr.Slider(-100, 100, 0, step=1, label="Blacks")

                    with gr.Accordion("Virtual Studio Relighting", open=False):
                        relight = gr.Slider(0, 100, 0, step=1, label="Relight Strength")
                        relight_azimuth = gr.Slider(-180, 180, 0, step=1, label="Light Azimuth")
                        relight_elevation = gr.Slider(-90, 90, 30, step=1, label="Light Elevation")

                    with gr.Accordion("Eyes & Lips", open=False):
                        eye_enhance = gr.Slider(0, 100, 5, step=1, label="Eye Enhance")
                        teeth_whiten = gr.Slider(0, 100, 5, step=1, label="Teeth Whiten")
                        lip_enhance = gr.Slider(0, 100, 5, step=1, label="Lip Enhance")
                        lip_tint = gr.Dropdown(choices=LIP_TINTS, value="none", label="Lip Tint")
                        blush = gr.Slider(0, 100, 0, step=1, label="Blush")

                    with gr.Accordion("Structure & Effects", open=False):
                        hair_enhance = gr.Slider(0, 100, 5, step=1, label="Hair Shine")
                        dodge_burn = gr.Slider(0, 100, 0, step=1, label="Dodge & Burn")
                        specular_bloom = gr.Slider(0, 100, 0, step=1, label="Specular Bloom")
                        bloom = gr.Slider(0, 100, 0, step=1, label="Bloom")
                        bloom_threshold = gr.Slider(150, 250, 210, step=1, label="Bloom Threshold")
                        bloom_softness = gr.Slider(1, 100, 30, step=1, label="Bloom Softness")

                    with gr.Accordion("Color Transfer", open=False):
                        gr.Markdown("Upload a reference image to match its colour tone")
                        color_ref_img = gr.File(label="Reference Image (RAW supported)", file_types=["image"])
                        color_ref_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Transfer Strength")

        with gr.Tab("Folder Automation & Ingestion"):
            with gr.Row():
                with gr.Column(scale=1):
                    folder_in = gr.Textbox(label="Input Folder Path", placeholder="/path/to/photos")
                    folder_out = gr.Textbox(label="Output Folder Path", placeholder="/path/to/exports")
                    
                    with gr.Group():
                        gr.Markdown("### Style Mode")
                        batch_style_type = gr.Radio(choices=["Use Standard Recipe", "Use Custom Style"], value="Use Standard Recipe", label="Style Mode")
                        batch_recipe = gr.Dropdown(choices=RECIPE_NAMES, value="natural", label="Standard Recipe")
                        batch_custom_style = gr.Dropdown(choices=custom_style_choices, label="Custom Style Profile")
                    
                    with gr.Group():
                        gr.Markdown("### Export Formatting")
                        batch_fmt = gr.Radio(choices=["JPEG", "PNG", "WebP"], value="JPEG", label="Format")
                        batch_quality = gr.Slider(10, 100, 95, step=1, label="Quality")
                        batch_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Export Resolution"
                        )
                        
                    with gr.Row():
                        auto_group_toggle = gr.Checkbox(label="Enable Rule-Based Auto-Grouping", value=True)
                        sheet_toggle = gr.Checkbox(label="Generate Contact Sheet", value=True)
                        zip_toggle = gr.Checkbox(label="Package into ZIP", value=True)
                        
                    batch_btn = gr.Button("Process Entire Folder", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    batch_sheet_out = gr.Image(label="Generated Contact Sheet", height=320)
                    batch_zip_out = gr.File(label="Download Packaged ZIP")
                    batch_status = gr.Textbox(label="Execution Log & Statistics", lines=12, interactive=False)

        with gr.Tab("Style Library & Learning"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Save Current Settings as Custom Style")
                    save_name = gr.Textbox(label="Style Name", placeholder="e.g. Dennis Cosplay v4")
                    save_author = gr.Textbox(label="Author", value="Dennis")
                    save_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="moody, cosplay, soft")
                    save_style_btn = gr.Button("Save Sliders to Style Library", variant="primary")
                    save_status = gr.Textbox(label="Save Status", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Style Dataset Learning (Pairs Extractor)")
                    learn_orig_dir = gr.Textbox(label="Original Folder Path", placeholder="/path/to/originals")
                    learn_edit_dir = gr.Textbox(label="Edited Folder Path", placeholder="/path/to/edited")
                    learn_name = gr.Textbox(label="Learned Style Name", placeholder="e.g. Dennis_Cosplay_V7")
                    learn_author = gr.Textbox(label="Author", value="Dennis")
                    learn_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="learned, cosplay")
                    learn_style_btn = gr.Button("Extract & Save Style", variant="primary")
                    learn_status = gr.Textbox(label="Learning Status", lines=5, interactive=False)

    recipe.change(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=[smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth,
                 whiten, equalize,
                 relight, relight_azimuth, relight_elevation,
                 eye_enhance, teeth_whiten,
                 lip_enhance, lip_tint, blush,
                 hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
                 highlights, shadows, whites, blacks],
    )

    custom_style_preset.change(
        fn=apply_custom_style,
        inputs=[custom_style_preset],
        outputs=[smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth,
                 whiten, equalize,
                 relight, relight_azimuth, relight_elevation,
                 eye_enhance, teeth_whiten,
                 lip_enhance, lip_tint, blush,
                 hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
                 highlights, shadows, whites, blacks],
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

    batch_btn.click(
        fn=on_process_folder,
        inputs=[folder_in, folder_out, batch_style_type, batch_custom_style, batch_recipe,
                batch_fmt, batch_quality, batch_res, auto_group_toggle, sheet_toggle, zip_toggle],
        outputs=[batch_sheet_out, batch_zip_out, batch_status]
    )

    process_btn.click(
        fn=process_image,
        inputs=[img_input, recipe,
                smooth, mid_reduction, texture_opacity, pore_synthesis, nose_smooth,
                whiten, equalize,
                relight, relight_azimuth, relight_elevation,
                eye_enhance, teeth_whiten,
                lip_enhance, lip_tint, blush,
                hair_enhance, dodge_burn, specular_bloom, bloom, bloom_threshold, bloom_softness, contrast, brightness,
                highlights, shadows, whites, blacks,
                color_ref_img, color_ref_strength,
                show_compare, fast,
                export_fmt, export_quality, export_res],
        outputs=[img_output, export_file, status],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
