#!/usr/bin/env python3

import sys
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine
from retouch.recipes import RECIPES
from cli import _imread_exif, _make_comparison

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
        "nose_smooth": 0,
        "whiten": int(rec["skin"].get("rosy", rec["skin"].get("porcelain", 0)) * 100),
        "equalize": int(rec["skin"].get("equalize", 0) * 100),
        "eye_enhance": int(rec["eyes"].get("iris", 0) * 100),
        "lip_enhance": int(rec["lips"].get("gloss", 0) * 100),
        "lip_tint": rec["lips"].get("tint", "none"),
        "blush": 25 if recipe_name in ("cosplay", "scifi_cosplay", "cyber_doll", "anime_cosplay", "anime", "xiaohongshu", "idol", "wedding") else 0,
        "teeth_whiten": int(rec["eyes"].get("whites", 0) * 100),
        "hair_enhance": int(rec["hair"].get("shine", 0) * 100),
        "dodge_burn": int(rec["dodge_burn"].get("amount", 0) * 100),
        "specular_bloom": rec.get("specular_bloom", 0),
        "contrast": rec.get("contrast", 0),
        "brightness": 0,
        "highlights": 0,
        "shadows": 0,
        "whites": 0,
        "blacks": 0,
    }


def process_image(img_path, recipe,
                  smooth, mid_reduction, texture_opacity, nose_smooth,
                  whiten, equalize,
                  eye_enhance, teeth_whiten,
                  lip_enhance, lip_tint, blush,
                  hair_enhance, dodge_burn, specular_bloom, contrast, brightness,
                  highlights, shadows, whites, blacks,
                  color_ref_path, color_ref_strength,
                  show_compare, fast,
                  export_fmt, export_quality, export_res):
    if img_path is None:
        return None, None, "Please upload an image first."

    try:
        if isinstance(img_path, dict):
            img_path = img_path.get("name") or img_path.get("path")
        img_bgr = _imread_exif(img_path)
        original = img_bgr.copy()

        lip_tint_val = lip_tint if lip_tint != "none" else None

        color_ref_bgr = None
        if color_ref_path is not None:
            if isinstance(color_ref_path, dict):
                color_ref_path = color_ref_path.get("name") or color_ref_path.get("path")
            color_ref_bgr = _imread_exif(color_ref_path)

        engine = get_engine()
        result = engine.process(
            img_bgr,
            recipe=recipe,
            smooth=smooth,
            mid_reduction=mid_reduction,
            texture_opacity=texture_opacity,
            nose_smooth=nose_smooth if nose_smooth > 0 else None,
            whiten=whiten,
            equalize=equalize,
            eye_enhance=eye_enhance,
            teeth_whiten=teeth_whiten,
            lip_enhance=lip_enhance,
            lip_tint=lip_tint_val,
            blush=blush,
            hair_enhance=hair_enhance,
            dodge_burn=dodge_burn,
            specular_bloom=specular_bloom,
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

        # Resize export to chosen max dimension
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
        out_path = os.path.join(tempfile.gettempdir(), f"retouch_export{ext}")
        write_params = []
        if export_fmt == "JPEG":
            write_params = [cv2.IMWRITE_JPEG_QUALITY, export_quality]
        elif export_fmt == "WebP":
            write_params = [cv2.IMWRITE_WEBP_QUALITY, export_quality]
        cv2.imwrite(out_path, export_img, write_params)

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
            return combined, out_path, "Done ✓"
        else:
            if result_rgb.shape[0] > 900:
                scale = 900 / result_rgb.shape[0]
                new_w = int(result_rgb.shape[1] * scale)
                result_rgb = cv2.resize(result_rgb, (new_w, 900), interpolation=cv2.INTER_AREA)
            return result_rgb, out_path, "Done ✓"

    except Exception as e:
        return None, None, f"Error: {e}"


def on_recipe_change(recipe):
    d = recipe_defaults(recipe)
    return (
        d["smooth"], d["mid_reduction"], d["texture_opacity"], d["nose_smooth"],
        d["whiten"], d["equalize"],
        d["eye_enhance"], d["teeth_whiten"],
        d["lip_enhance"], d["lip_tint"], d["blush"],
        d["hair_enhance"], d["dodge_burn"], d["specular_bloom"], d["contrast"], d["brightness"],
        d["highlights"], d["shadows"], d["whites"], d["blacks"],
    )


LIP_TINTS = ["none", "cosplay", "rose", "pink", "coral", "natural", "berry"]

with gr.Blocks(title="Retouch GUI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Retouch — AI Face Retouching Tool")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.File(label="Input Image (RAW supported)", file_types=["image"])

            with gr.Group():
                recipe = gr.Dropdown(
                    choices=RECIPE_NAMES, value="natural", label="Recipe",
                    info="Select a recipe to auto-fill sliders"
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

            with gr.Accordion("Color Transfer", open=False):
                gr.Markdown("Upload a reference image to match its colour tone")
                color_ref_img = gr.File(label="Reference Image (RAW supported)", file_types=["image"])
                color_ref_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Transfer Strength")

    recipe.change(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=[smooth, mid_reduction, texture_opacity, nose_smooth,
                 whiten, equalize,
                 eye_enhance, teeth_whiten,
                 lip_enhance, lip_tint, blush,
                 hair_enhance, dodge_burn, specular_bloom, contrast, brightness,
                 highlights, shadows, whites, blacks],
    )

    process_btn.click(
        fn=process_image,
        inputs=[img_input, recipe,
                smooth, mid_reduction, texture_opacity, nose_smooth,
                whiten, equalize,
                eye_enhance, teeth_whiten,
                lip_enhance, lip_tint, blush,
                hair_enhance, dodge_burn, specular_bloom, contrast, brightness,
                highlights, shadows, whites, blacks,
                color_ref_img, color_ref_strength,
                show_compare, fast,
                export_fmt, export_quality, export_res],
        outputs=[img_output, export_file, status],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
