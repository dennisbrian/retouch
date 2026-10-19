"""Session-owned layer editor in the existing Gradio desktop shell."""
import shutil
import io
import math
import tempfile
import uuid
import threading
import time
import weakref
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageCms, ImageOps

from .adjustment_controls import controls_adjustment, curve_graph, curve_tables
from .adjustments import CHANNELS, KINDS
from .commands import DocumentHistory
from .composite import RENDER_BLEND_MODES, composite, export_jpeg, export_png
from .document import Document, Layer, MAX_SURFACE_PIXELS
from .project_store import load_project
from .recipe_state import EXTRA_RECIPE_CONTROLS, RECIPE_OVERRIDE_SPECS
from .recovery import Autosaver, discard_recovery, list_recoveries, load_recovery

PREVIEW_SIDE = 1024

# Tier-1 hotkey parity with the native Compositor (KeyboardShortcuts.swift):
# cmd/ctrl+Z undo, cmd/ctrl+shift+Z and cmd/ctrl+Y redo, cmd/ctrl+D clear
# selection. Gradio 4.44 accepts `js` only on the root Blocks, so gui.py
# passes this constant to its shell's Blocks(js=...). Keeping it
# module-level means the wiring and the source-level test reference one
# symbol.
EDITOR_HOTKEYS_JS = """() => {
    document.addEventListener('keydown', (e) => {
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
            || t.tagName === 'SELECT' || t.isContentEditable)) return;
        const M = e.metaKey || e.ctrlKey;
        if (!M) return;
        const S = e.shiftKey;
        const k = e.key.toLowerCase();
        const hit = (id) => {
            const btn = document.getElementById(id);
            if (btn) { e.preventDefault(); btn.click(); }
        };
        if (M && S && k === 'z') return hit('editor-redo');
        if (M && k === 'z') return hit('editor-undo');
        if (M && k === 'y' && !S) return hit('editor-redo');
        if (M && k === 'd') return hit('editor-clear-selection');
    });
}"""


def _photo(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_SURFACE_PIXELS:
            raise ValueError('Photo exceeds the editor pixel budget')
        pixels = ImageOps.exif_transpose(image).convert('RGBA')
        profile = image.info.get('icc_profile')
        if profile:
            # The document format declares sRGB; convert tagged sources first.
            try:
                pixels = ImageCms.profileToProfile(
                    pixels, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                    ImageCms.createProfile('sRGB'), outputMode='RGBA')
            except ImageCms.PyCMSError as exc:
                raise ValueError('Could not convert the photo ICC profile to sRGB') from exc
        return np.asarray(pixels)


def preview_document(document):
    """Render resized layer copies; native buffers remain unchanged."""
    scale = min(1.0, PREVIEW_SIDE / max(document.width, document.height))
    if scale == 1:
        return composite(document)
    preview = copy(document)
    preview.width = max(1, round(document.width * scale))
    preview.height = max(1, round(document.height * scale))
    preview.layers = []
    for original in document.layers:
        layer = copy(original)
        layer.recipe_source = layer.recipe_settings = None  # preview-only raster copies
        layer.origin = tuple(round(v * scale) for v in original.origin)
        layer.size = tuple(max(1, round(v * scale)) for v in original.size)
        if original.mask_placement is not None:
            # An unlinked placement is document-space; it must scale with the
            # preview like the layer itself (upstream ImageResizer.swift:97).
            p = original.mask_placement
            layer.mask_placement = {**p,
                'origin': tuple(round(v * scale) for v in p['origin']),
                'size': tuple(max(1, round(v * scale)) for v in p['size'])}
        if original.pixels is not None:
            # Pillow interpolates RGBA in premultiplied form.
            layer.pixels = _preview_scaled(
                original.pixels, layer.size, _preview_lanczos)
        if original.mask is not None:
            layer.mask = _preview_scaled(original.mask, layer.size, _preview_mask)
        preview.layers.append(layer)
    return composite(preview)


def _preview_lanczos(array, size):
    return np.asarray(Image.fromarray(array).resize(size, Image.Resampling.LANCZOS))


def _preview_mask(array, size):
    return cv2.resize(array, size, interpolation=cv2.INTER_LINEAR)


# Preview resampling is the interactive hot path: every edit re-renders the
# whole document, but an unchanged layer's frozen source buffer is shared
# across snapshots (commands._snapshot keeps pixel identity), so its resized
# copy is reusable. Key on the buffer object itself plus the geometry that
# changes the result. Keep only a weak reference to the source: cached previews
# must not pin full-resolution undo buffers or defeat DocumentHistory's budget.
_PREVIEW_CACHE = {}
_PREVIEW_CACHE_LIMIT = 64


def _preview_scaled(source, size, resize):
    """Resized preview of ``source``, memoised while the source is unchanged."""
    key = (id(source), source.shape, tuple(size))
    cached = _PREVIEW_CACHE.get(key)
    if cached is None or cached[0]() is not source:
        cached = (weakref.ref(source), resize(source, size))
        if len(_PREVIEW_CACHE) >= _PREVIEW_CACHE_LIMIT:
            _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))
        _PREVIEW_CACHE[key] = cached
    return cached[1]


def _clear_preview_cache():
    _PREVIEW_CACHE.clear()


def has_strokes(canvas):
    return bool(canvas and any(np.asarray(layer)[..., 3].any()
                              for layer in canvas.get('layers', [])))


@dataclass
class EditorSession:
    history: DocumentHistory
    selected_id: str
    workspace: str
    job: object = None
    autosave: object = None
    selection: object = None

    @classmethod
    def create(cls, document, saved=False, title='', recovery_id=None, recovery_root=None):
        return cls(DocumentHistory(document, saved=saved),
                   document.active_layer_id or (document.layers[-1].id if document.layers else None),
                   tempfile.mkdtemp(prefix='retouch-layer-editor-'),
                   autosave=Autosaver(title, recovery_root, recovery_id))

    def checkpoint(self):
        """Autosave unsaved work for crash recovery (background, never raises)."""
        if self.autosave is not None:
            self.autosave.checkpoint(self.history)

    def close(self, keep_recovery=False):
        """Release the session. ``keep_recovery`` leaves unsaved work offered
        back on the next start (window closed); otherwise it is deleted."""
        if self.autosave is not None:
            self.autosave.close(keep=keep_recovery)
        if self.job and self.job.status in ('queued', 'running', 'timed out'):
            self.job.cancel()
            if self.job.status == 'queued':
                self.job.status, self.job.finished = 'cancelled', True
            def cleanup():
                while not self.job.finished:
                    time.sleep(0.25)
                shutil.rmtree(self.workspace, ignore_errors=True)
            threading.Thread(target=cleanup, daemon=True).start()
        else:
            shutil.rmtree(self.workspace, ignore_errors=True)

    def selected(self):
        doc = self.history.document
        if self.selected_id not in {layer.id for layer in doc.layers}:
            self.selected_id = doc.active_layer_id
        return doc.layer(self.selected_id) if self.selected_id else None

    def apply_mask(self, canvas, action=None):
        if action not in (None, 'Reveal', 'Hide'):
            raise ValueError('Choose Reveal or Hide for mask strokes')
        layer = self.selected()
        if layer is None:
            raise ValueError('Select a layer first')
        width, height = ((layer.pixels.shape[1], layer.pixels.shape[0])
                         if layer.pixels is not None else layer.size)
        if width * height > MAX_SURFACE_PIXELS or max(width, height) > 30000:
            raise ValueError('Layer mask exceeds the surface budget')
        placement = layer.mask_placement
        if placement is not None:
            # The brush canvas shows the layer's rectangle, but an unlinked mask
            # lives in its own grid: warp each stroke's coverage from layer
            # coordinates into mask coordinates (upstream BrushStroke keeps the
            # mask's placement as the stroke's base transform for the same
            # reason). Strokes clamp to the mask's own rect.
            from .composite import _doc_matrix
            mw, mh = (int(round(placement['size'][0])), int(round(placement['size'][1])))
            mask = (np.full((mh, mw), 255, np.uint8) if layer.mask is None
                    else cv2.resize(layer.mask, (mw, mh), interpolation=cv2.INTER_LINEAR))
            m_doc = _doc_matrix(placement['origin'], placement['size'],
                                placement['rotation'], placement['flipX'], placement['flipY'])
            l_doc = _doc_matrix(layer.origin, layer.size, layer.rotation,
                                layer.flip_x, layer.flip_y)
            # Layer-grid stroke pixel -> mask pixel (doc round trip).
            to_mask = (np.linalg.inv(np.vstack([m_doc, [0, 0, 1]]))
                       @ np.vstack([l_doc, [0, 0, 1]]))[:2].astype(np.float32)
            border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
            background = 255.0 if border.mean() >= 127.5 else 0.0
        else:
            mask = (np.full((height, width), 255, np.uint8) if layer.mask is None
                    else cv2.resize(layer.mask, (width, height), interpolation=cv2.INTER_LINEAR))
        painted = False
        for stroke in (canvas or {}).get('layers', []):
            stroke = np.asarray(stroke)
            if stroke.dtype != np.uint8 or stroke.ndim != 3 or stroke.shape[2] != 4:
                raise ValueError('Invalid brush data')
            if not stroke[..., 3].any():
                continue
            rgba = np.asarray(Image.fromarray(stroke).resize(
                (width, height), Image.Resampling.BILINEAR)).astype(np.float32)
            # The brush canvas shows the layer's orientation; store coverage
            # in source coordinates so the compositor flips pixels and mask once.
            if layer.flip_y:
                rgba = rgba[::-1]
            if layer.flip_x:
                rgba = rgba[:, ::-1]
            alpha = rgba[..., 3] / 255.0
            if placement is not None:
                alpha = cv2.warpAffine(alpha.astype(np.float32), to_mask, (mw, mh),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0.0)
                if not alpha.any():
                    continue  # stroke missed the mask's rectangle entirely
            # Explicit action avoids Gradio palette resets changing mask intent.
            # The default preserves color-based callers and existing saved workflows.
            value = (rgba[..., :3].mean(axis=2) if action is None
                     else np.full(alpha.shape, 255.0 if action == 'Reveal' else 0.0,
                                  np.float32))
            if placement is not None:
                value = cv2.warpAffine(value, to_mask, (mw, mh), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=background)
            mask = np.rint(mask * (1 - alpha) + value * alpha).astype(np.uint8)
            painted = True
        if painted:
            self.history.set_mask(layer.id, mask, layer.mask_enabled, placement)
        return painted


    def stroke_coverage(self, canvas, layer):
        """Unapplied brush strokes as 8-bit coverage over the layer's own pixels."""
        height, width = layer.pixels.shape[:2]
        coverage = np.zeros((height, width), np.float32)
        for stroke in (canvas or {}).get('layers', []):
            stroke = np.asarray(stroke)
            if stroke.dtype != np.uint8 or stroke.ndim != 3 or stroke.shape[2] != 4:
                raise ValueError('Invalid brush data')
            if not stroke[..., 3].any():
                continue
            alpha = cv2.resize(stroke[..., 3].astype(np.float32), (width, height),
                               interpolation=cv2.INTER_LINEAR)
            # The canvas shows the layer flipped like the composite; pixels are stored unflipped.
            if layer.flip_y:
                alpha = alpha[::-1]
            if layer.flip_x:
                alpha = alpha[:, ::-1]
            coverage = np.maximum(coverage, alpha)
        return np.rint(coverage).astype(np.uint8)

    def paint_pixels(self, canvas, tool, heal_mode='Content-Aware', opacity=100,
                     offset=(0, 0), softness=0):
        """Spot Healing or Clone Stamp the brushed area of the selected pixel layer."""
        from .healing import clone_stamp, spot_heal
        layer = self.selected()
        if layer is None:
            raise ValueError('Select a layer first')
        if layer.pixels is None or layer.is_group or layer.adjustment is not None:
            raise ValueError('Healing and cloning need a photo or pixel layer')
        for value in (opacity, softness, *offset):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ValueError('Healing and clone settings must be finite numbers')
        coverage = self.stroke_coverage(canvas, layer)
        if not coverage.any():
            raise ValueError('Brush over the area first')
        if softness < 0:
            raise ValueError('Edge softness cannot be negative')
        if softness > 0:
            # The canvas brush is hard; a soft tip (native Clone Stamp's default)
            # fades the copy in over about ``softness`` layer pixels.
            coverage = np.rint(cv2.GaussianBlur(coverage.astype(np.float32), (0, 0),
                                                softness / 2)).astype(np.uint8)
        if tool == 'heal':
            pixels = spot_heal(layer.pixels, coverage, heal_mode, opacity / 100,
                               seed=self.history.revision)
            label = 'Spot Healing'
        elif tool == 'clone':
            pixels = clone_stamp(layer.pixels, coverage,
                                 (int(round(offset[0])), int(round(offset[1]))), opacity / 100)
            label = 'Clone Stamp'
        else:
            raise ValueError('Unknown pixel tool')
        if np.array_equal(pixels, layer.pixels):
            return False
        return self.history.replace_pixels(layer.id, pixels, label=label)


def _view(session, message=''):
    if session is None:
        return (None, None, gr.update(choices=[], value=None), '', True, 100,
                True, None, gr.update(interactive=False), gr.update(interactive=False),
                'Open a photo or project to start.', gr.update(choices=list(RENDER_BLEND_MODES), value='Normal'),
                0, 0, False, False, 1, 1, *_tool_view(None))
    # Every edit path renders through here, so this is the one autosave hook.
    session.checkpoint()
    doc = session.history.document
    layer = session.selected()
    canvas = None
    if layer is not None:
        scale = min(1, PREVIEW_SIDE / max(layer.size))
        size = tuple(max(1, round(v * scale)) for v in layer.size)
        image = (np.asarray(Image.fromarray(layer.pixels).resize(size)) if layer.pixels is not None
                 else np.full((size[1], size[0], 4), 255, np.uint8))
        if layer.flip_y:
            image = image[::-1]
        if layer.flip_x:
            image = image[:, ::-1]
        canvas = {'background': image, 'layers': [], 'composite': image}
    state = 'Unsaved changes' if session.history.dirty else 'Saved'
    return (session, preview_document(doc),
            gr.update(choices=[(('  ' * len(doc.ancestors(l))) + ('Folder: ' if l.is_group else '') + l.name, l.id)
                               for l in reversed(doc.ordered_layers())],
                      value=session.selected_id),
            layer.name if layer else '', layer.visible if layer else True,
            layer.opacity * 100 if layer else 100,
            layer.mask_enabled if layer else True, canvas,
            gr.update(interactive=session.history.can_undo),
            gr.update(interactive=session.history.can_redo),
            f'{state} · {doc.width} × {doc.height} · {len(doc.layers)} layers. {message}',
            gr.update(choices=list(RENDER_BLEND_MODES) + (
                [layer.blend_mode] if layer and layer.blend_mode not in RENDER_BLEND_MODES else []),
                value=layer.blend_mode if layer else 'Normal'),
            layer.origin[0] if layer else 0, layer.origin[1] if layer else 0,
            layer.flip_x if layer else False, layer.flip_y if layer else False,
            layer.size[0] if layer else 1, layer.size[1] if layer else 1, *_tool_view(session))


def _tool_view(session):
    doc = session.history.document if session else None
    layer = session.selected() if session else None
    folders = [('Canvas', '')] + ([(l.name, l.id) for l in doc.layers if l.is_group] if doc else [])
    sources = [('None', '')] + ([(l.name, l.id) for l in doc.layers
                               if not l.is_group and l.adjustment is None and (layer is None or l.id != layer.id)] if doc else [])
    selected = session.selection if session else None
    if selected is not None:
        factor = min(1, PREVIEW_SIDE / max(selected.shape))
        selected = cv2.resize(selected, (max(1, round(selected.shape[1] * factor)),
                                        max(1, round(selected.shape[0] * factor))))
    settings = layer.adjustment if layer and layer.adjustment else {}
    exposure = settings.get('exposureSettings') or {}
    ranges = (settings.get('levels') or {}).get('ranges') or [dict(black=0, white=255, gamma=1) for _ in range(4)]
    saturation = (settings.get('saturationSettings') or {}).get('saturation', 0)
    vibrance = (settings.get('vibranceSettings') or {}).get('vibrance', 0)
    temperature = (settings.get('temperatureSettings') or {}).get('temperature', 0)
    tint = (settings.get('tintSettings') or {}).get('tint', 0)
    brightness = (settings.get('brightnessSettings') or {}).get('brightness', 0)
    contrast = (settings.get('contrastSettings') or {}).get('contrast', 0)
    hue = (settings.get('hueSettings') or {}).get('hue', 0)
    tables = curve_tables(settings)
    grain = settings.get('grainSettings') or {}
    seed = grain.get('seed', 0) if settings.get('kind') == 'Grain' else settings.get('noiseSeed', 0)
    effects = layer.effects if layer and layer.effects else None
    shadow = effects.shadow if effects and effects.shadow else None
    overlay = effects.colorOverlay if effects and effects.colorOverlay else None
    inner_s = effects.innerShadow if effects and effects.innerShadow else None
    glow_o = effects.outerGlow if effects and effects.outerGlow else None
    glow_i = effects.innerGlow if effects and effects.innerGlow else None
    stroke = effects.stroke if effects and effects.stroke else None
    return (layer.rotation if layer else 0,
            gr.update(choices=folders, value=(layer.parent_id or '') if layer else ''),
            gr.update(choices=sources, value=(layer.mask_source_id or '') if layer else ''),
            gr.update(choices=[(l.name, l.id) for l in doc.ordered_layers()] if doc else [], value=[]),
            selected, settings.get('kind', 'Exposure'), exposure.get('exposure', 0),
            exposure.get('offset', 0), exposure.get('gamma', 1) if settings.get('kind') == 'Exposure' else ranges[0]['gamma'],
            ranges[0]['black'], ranges[0]['white'], ranges[1]['gamma'], ranges[2]['gamma'], ranges[3]['gamma'],
            saturation, vibrance, temperature, tint, brightness, contrast, hue,
            stroke is not None, stroke.size if stroke else 4, stroke.inside if stroke else False,
            round((stroke.opacity if stroke else 1) * 100),
            gr.update(value=_hex_color(stroke.red, stroke.green, stroke.blue) if stroke else '#000000',
                      visible=stroke is not None),
            shadow is not None, shadow.angle if shadow else 90, shadow.distance if shadow else 20,
            shadow.blur if shadow else 20, round((shadow.opacity if shadow else 0.5) * 100),
            gr.update(value=_hex_color(shadow.red, shadow.green, shadow.blue) if shadow else '#000000',
                      visible=shadow is not None),
    overlay is not None, round((overlay.opacity if overlay else 1) * 100),
            gr.update(value=_hex_color(overlay.red, overlay.green, overlay.blue) if overlay else '#000000',
                      visible=overlay is not None),
            inner_s is not None, inner_s.angle if inner_s else 90,
            inner_s.distance if inner_s else 10, inner_s.blur if inner_s else 10,
            round((inner_s.opacity if inner_s else 0.5) * 100),
            gr.update(value=_hex_color(inner_s.red, inner_s.green, inner_s.blue) if inner_s else '#000000',
                      visible=inner_s is not None),
            glow_o is not None, glow_o.size if glow_o else 20,
            round((glow_o.opacity if glow_o else 0.75) * 100),
            gr.update(value=_hex_color(glow_o.red, glow_o.green, glow_o.blue) if glow_o else '#ffffff',
                      visible=glow_o is not None),
            glow_i is not None, glow_i.size if glow_i else 10,
            round((glow_i.opacity if glow_i else 0.75) * 100),
            gr.update(value=_hex_color(glow_i.red, glow_i.green, glow_i.blue) if glow_i else '#ffffff',
                      visible=glow_i is not None),
            *tables, grain.get('amount', 25), grain.get('size', 1.5),
            grain.get('roughness', 50), seed, settings.get('noiseAmount', 25),
            settings.get('noiseGaussian', False), settings.get('noiseMonochromatic', False),
            curve_graph(*tables))


def _hex_color(red, green, blue):
    """A 0..1 RGB triple as ``#rrggbb`` for a colour picker."""
    return '#%02x%02x%02x' % tuple(int(round(min(1.0, max(0.0, v)) * 255)) for v in (red, green, blue))


def _rgb(hex_color):
    """``#rrggbb`` (or a Gradio colour value) into a 0..1 RGB triple."""
    text = str(hex_color or '#000000').lstrip('#')
    if len(text) == 3:
        text = ''.join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError('Choose a shadow and overlay colour as #rrggbb')
    try:
        red, green, blue = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError('Choose a shadow and overlay colour as #rrggbb') from exc
    return red, green, blue


def build_editor_tab():
    """Add a complete independent tab inside the application's Tabs block."""
    with gr.Tab('Layer Editor') as tab:
        session = gr.State(None, delete_callback=lambda value: value.close(keep_recovery=True)
                           if value else None)
        gr.Markdown('Open a photo or local `.comp` folder. Choose Reveal or Hide for mask strokes. '
                    'Apply brush strokes before changing layers. Export keeps native resolution.')
        with gr.Row():
            with gr.Column():
                photo = gr.File(label='Photo', file_types=['image'], type='filepath', height=100)
                open_photo = gr.Button('Open photo')
            with gr.Column():
                project_path = gr.Textbox(label='Project folder', placeholder='/path/to/photo.comp',
                                          lines=1, max_lines=1)
                open_project = gr.Button('Open project')
        with gr.Accordion('Recover unsaved work', open=False) as recovery_panel:
            gr.Markdown('Unsaved edits are autosaved while you work. If the app quits '
                        'or the window closes before you save, restore them here. '
                        'Autosaves are deleted when you save, discard them, or after 30 days.')
            recoveries = gr.Dropdown(label='Autosaved documents', choices=[])
            with gr.Row():
                refresh_recoveries = gr.Button('Refresh list')
                restore = gr.Button('Restore')
                discard_saved = gr.Button('Discard autosave')
        discard = gr.Checkbox(label='Discard unsaved edits and unapplied strokes when opening', value=False)
        with gr.Row():
            with gr.Column(scale=2):
                preview = gr.Image(label='Composite preview', interactive=False, image_mode='RGBA')
                mask_action = gr.Radio(['Reveal', 'Hide'], value='Hide', label='Mask brush action')
                canvas = gr.ImageEditor(label='Paint selected layer mask', type='numpy',
                                        image_mode='RGBA', sources=(), transforms=(),
                                        format='png',
                                        brush=gr.Brush(colors=['#ff3366'],
                                                       color_mode='fixed', default_color='#ff3366'),
                                        height=450)
                gr.Markdown('The canvas shows the layer before its mask. Colored strokes mark where '
                            'the chosen Reveal or Hide action applies. The eraser removes unapplied strokes. '
                            'Choose the action for these strokes before applying.')
                apply_mask = gr.Button('Apply mask strokes')
                with gr.Accordion('Spot healing & clone stamp', open=False):
                    gr.Markdown('Brush over a spot on the canvas above, then heal or clone instead '
                                'of applying a mask. These change the selected layer\'s own pixels '
                                '(at full resolution) as one undo step; nothing outside the strokes changes.')
                    heal_mode = gr.Radio(['Content-Aware', 'Create Texture', 'Proximity Match'],
                                         value='Content-Aware', label='Healing mode')
                    heal_opacity = gr.Slider(1, 100, value=100, step=1, label='Opacity (%)')
                    heal = gr.Button('Spot heal strokes')
                    gr.Markdown('Content-Aware copies texture from the nearby patch that best matches '
                                'the edge of the spot; Proximity Match takes the closest good patch; '
                                'Create Texture fills smoothly from the edges and adds matching grain.')
                    with gr.Row():
                        clone_dx = gr.Number(label='Clone from X offset (layer pixels)', value=0, precision=0)
                        clone_dy = gr.Number(label='Clone from Y offset (layer pixels)', value=0, precision=0)
                    clone_softness = gr.Slider(0, 200, value=20, step=1,
                                               label='Clone edge softness (layer pixels)')
                    clone = gr.Button('Clone stamp strokes')
                    gr.Markdown('Each brushed pixel copies the layer pixel that far away (positive X '
                                'is right, positive Y is down, in the layer\'s unflipped pixels). '
                                'The offset stays until you change it, like aligned Clone Stamp.')
            with gr.Column(scale=1):
                layers = gr.Dropdown(label='Layers (top first)', choices=[])
                name = gr.Textbox(label='Layer name')
                visible = gr.Checkbox(label='Visible', value=True)
                opacity = gr.Slider(0, 100, value=100, step=1, label='Opacity')
                mask_enabled = gr.Checkbox(label='Mask enabled', value=True)
                blend = gr.Dropdown(label='Blend mode', choices=list(RENDER_BLEND_MODES), value='Normal')
                properties = gr.Button('Apply layer properties')
                with gr.Row():
                    duplicate = gr.Button('Duplicate layer')
                    invert_mask = gr.Button('Invert mask')
                    reset_mask = gr.Button('Reset mask')
                gr.Markdown('Invert mask swaps revealed and hidden areas. '
                            'A layer without a mask becomes fully masked out; '
                            'enable its mask to see the change. Reset mask removes '
                            'painted coverage and reveals the layer. Both can be undone.')
                with gr.Accordion('Position, rotation & flips', open=False):
                    gr.Markdown('Position uses canvas pixels from the top-left. '
                                'Negative positions are allowed; pixels outside the canvas are clipped. '
                                'Flips move the mask with the layer.')
                    with gr.Row():
                        position_x = gr.Number(label='X position', value=0, precision=0)
                        position_y = gr.Number(label='Y position', value=0, precision=0)
                    flip_x = gr.Checkbox(label='Flip horizontally', value=False)
                    flip_y = gr.Checkbox(label='Flip vertically', value=False)
                    rotation = gr.Number(label='Clockwise rotation (degrees)', value=0)
                    transform = gr.Button('Apply transform')
                    gr.Markdown('Rotation uses the layer center. The mask brush shows the unrotated layer; '
                                'its strokes rotate with the pixels in the composite.')
                    with gr.Row():
                        layer_width = gr.Number(label='Layer width (pixels)', value=1, precision=0)
                        layer_height = gr.Number(label='Layer height (pixels)', value=1, precision=0)
                    keep_aspect = gr.Checkbox(label='Keep aspect ratio', value=True)
                    resize_axis = gr.Radio(['Width', 'Height'], value='Width', label='Resize from')
                    gr.Markdown('With Keep aspect ratio on, enter the dimension selected '
                                'in Resize from; the other is calculated from the current '
                                'layer proportions. Turn it off to set both dimensions freely. '
                                'Original pixels are preserved.')
                    resize = gr.Button('Apply layer size')
                    fit = gr.Button('Fit layer to canvas')
                    gr.Markdown('Fit scales the current layer proportions to fit inside '
                                'the canvas and centers it. Source pixels and masks are kept; '
                                'size and position change together in one undo step.')
                with gr.Accordion('Image & canvas size', open=False):
                    with gr.Row():
                        crop_x = gr.Number(label='Crop X', value=0, precision=0)
                        crop_y = gr.Number(label='Crop Y', value=0, precision=0)
                    with gr.Row():
                        crop_width = gr.Number(label='Crop width (pixels)', value=100, precision=0)
                        crop_height = gr.Number(label='Crop height (pixels)', value=100, precision=0)
                    crop_canvas = gr.Button('Crop canvas')
                    crop_selection = gr.Button('Crop to selection bbox')
                    gr.Markdown('Crops re-square the canvas; layers, masks and guides keep '
                                'their pixels and shift together. Cropping clears the '
                                'selection.')
                with gr.Accordion('Folders & clipping', open=False):
                    parent_folder = gr.Dropdown(label='Parent folder', choices=[('Canvas', '')], value='')
                    clipping_source = gr.Dropdown(label='Clip to layer alpha', choices=[('None', '')], value='')
                    links = gr.Button('Apply folder & clipping')
                    group_members = gr.Dropdown(label='Adjacent layers to group', choices=[], multiselect=True)
                    group = gr.Button('Group selected layers')
                    ungroup = gr.Button('Ungroup selected folder')
                    gr.Markdown('Folders pass through: opacity and masks affect each child. '
                                'Removing a folder removes its children. Duplicate copies the whole folder. '
                                'Adjacent clipped siblings share the base alpha; other links act as live alpha masks.')
                with gr.Accordion('Selections', open=False):
                    selection_kind = gr.Radio(['Rectangle', 'Ellipse', 'Polygon'], value='Rectangle', label='Selection shape')
                    with gr.Row():
                        selection_x = gr.Number(label='Selection X', value=0)
                        selection_y = gr.Number(label='Selection Y', value=0)
                    with gr.Row():
                        selection_width = gr.Number(label='Selection width', value=100)
                        selection_height = gr.Number(label='Selection height', value=100)
                    polygon = gr.Textbox(label='Polygon points [x, y]', value='[[0,0],[100,0],[50,100]]')
                    feather = gr.Slider(0, 250, value=0, step=1, label='Feather (canvas pixels)')
                    selection_mode = gr.Radio(['Replace', 'Add', 'Subtract', 'Intersect'], value='Replace', label='Combine selection')
                    create_selection = gr.Button('Create selection')
                    selection_preview = gr.Image(label='Selection coverage (white selected)', interactive=False, image_mode='L')
                    with gr.Row():
                        invert_selection = gr.Button('Invert selection')
                        clear_selection = gr.Button('Clear selection', elem_id='editor-clear-selection')
                    selection_amount = gr.Number(label='Expand / contract / border (px)', value=10, precision=0)
                    expand_selection = gr.Button('Expand selection')
                    contract_selection = gr.Button('Contract selection')
                    border_selection = gr.Button('Border selection')
                    with gr.Row():
                        move_dx = gr.Number(label='Move dx', value=0)
                        move_dy = gr.Number(label='Move dy', value=0)
                    move_selection = gr.Button('Move selection')
                    with gr.Row():
                        wand_x = gr.Number(label='Wand X', value=0, precision=0)
                        wand_y = gr.Number(label='Wand Y', value=0, precision=0)
                    wand_tolerance = gr.Slider(0, 255, value=32, step=1, label='Wand tolerance')
                    wand_sample = gr.Radio(['Point', '3x3', '5x5'], value='Point', label='Wand sample size')
                    wand_contiguous = gr.Checkbox(label='Contiguous', value=True)
                    wand_button = gr.Button('Magic wand selection')
                    from .semantic_selection import REGIONS
                    semantic_region = gr.Dropdown(list(REGIONS), value='Skin', label='Automatic selection region')
                    auto_select = gr.Button('Create automatic selection')
                    gr.Markdown('Selects from the visible opaque composite, across all detected faces. '
                                'Uses the Feather and Combine selection settings above. '
                                'Masks are estimated at up to 2048 pixels and scaled to the canvas. '
                                'Skin selects facial skin. Check the coverage preview, apply it to a layer mask, then refine '
                                'with Reveal/Hide strokes. Hair and eye boundaries may need correction.')
                    mask_mode = gr.Radio(['Replace', 'Add', 'Subtract', 'Intersect'], value='Replace', label='Combine with layer mask')
                    mask_selection = gr.Button('Apply selection to selected layer mask')
                    gr.Markdown('Selections use canvas coordinates and remain temporary. '
                                'Apply to a layer or folder mask to save them; this is one undo step.')
                with gr.Accordion('Grading adjustment layers', open=False):
                    grade_kind = gr.Radio(list(KINDS),
                                          value='Exposure', label='Adjustment')
                    grade_exposure = gr.Slider(-20, 20, value=0, step=.1, label='Exposure (stops)')
                    grade_offset = gr.Slider(-.5, .5, value=0, step=.01, label='Exposure offset (linear light)')
                    grade_gamma = gr.Slider(.1, 9.99, value=1, step=.01, label='Gamma')
                    grade_black = gr.Slider(0, 254, value=0, step=1, label='Levels black point')
                    grade_white = gr.Slider(1, 255, value=255, step=1, label='Levels white point')
                    grade_red = gr.Slider(.1, 9.99, value=1, step=.01, label='Red gamma')
                    grade_green = gr.Slider(.1, 9.99, value=1, step=.01, label='Green gamma')
                    grade_blue = gr.Slider(.1, 9.99, value=1, step=.01, label='Blue gamma')
                    grade_saturation = gr.Slider(-100, 100, value=0, step=1, label='Saturation')
                    grade_vibrance = gr.Slider(-100, 100, value=0, step=1, label='Vibrance')
                    grade_temperature = gr.Slider(-100, 100, value=0, step=1, label='Temperature')
                    grade_tint = gr.Slider(-100, 100, value=0, step=1, label='Tint')
                    grade_brightness = gr.Slider(-100, 100, value=0, step=1, label='Brightness')
                    grade_contrast = gr.Slider(-100, 100, value=0, step=1, label='Contrast')
                    grade_hue = gr.Slider(-180, 180, value=0, step=1, label='Hue (degrees)')
                    with gr.Group(visible=False) as curves_controls:
                        gr.Markdown('Edit input/output points (0–255). Keep the first input at 0 '
                                    'and the last at 255; add up to 32 rows in increasing input order. '
                                    'RGB applies to all channels. The graph previews the curves; '
                                    'click Add or Update to apply them to the image.')
                        curve_inputs = []
                        with gr.Tabs():
                            for channel in CHANNELS:
                                with gr.Tab(channel):
                                    curve_inputs.append(gr.Dataframe(
                                        value=[[0, 0], [255, 255]], headers=['Input', 'Output'],
                                        datatype=['number', 'number'], type='array',
                                        row_count=(2, 'dynamic'), col_count=(2, 'fixed'),
                                        interactive=True, label=channel + ' curve'))
                        curves_plot = gr.HTML(curve_graph(*curve_tables({})))
                        reset_curves = gr.Button('Reset all curves to identity')
                    with gr.Group(visible=False) as grain_controls:
                        grain_amount = gr.Slider(0, 100, value=25, step=.1, label='Grain amount')
                        grain_size = gr.Slider(.5, 20, value=1.5, step=.1, label='Grain size')
                        grain_roughness = gr.Slider(0, 100, value=50, step=.1, label='Grain roughness')
                    with gr.Group(visible=False) as noise_controls:
                        noise_amount = gr.Slider(.1, 400, value=25, step=.1, label='Noise amount')
                        noise_gaussian = gr.Checkbox(label='Gaussian noise', value=False)
                        noise_monochromatic = gr.Checkbox(label='Monochromatic noise', value=False)
                    with gr.Group(visible=False) as seed_controls:
                        texture_seed = gr.Number(label='Grain / noise seed', value=0, precision=0,
                                                 minimum=0, maximum=4294967295)
                        gr.Markdown('The same seed keeps the texture repeatable after saving and reopening.')
                    add_grade = gr.Button('Add adjustment layer')
                    update_grade = gr.Button('Update selected adjustment')
                    gr.Markdown('Adjustments change the layers below and preserve alpha. '
                                'Opacity and masks control coverage. Exposure uses stops, offset and gamma; '
                                'Levels uses black/white points and RGB/channel gamma. Other kinds expose one dedicated slider.')
                with gr.Accordion('Layer effects (raster layers)', open=False):
                    gr.Markdown('Stroke, drop shadow and colour overlay are kept with the layer '
                                'and apply to everything it shows. Distance, blur and size are in '
                                'layer pixels; angle is where the light comes from '
                                '(90 is straight above).')
                    effect_stroke = gr.Checkbox(label='Stroke', value=False)
                    with gr.Row():
                        stroke_size = gr.Slider(0, 200, value=4, step=1,
                                                label='Stroke size (pixels)', visible=False)
                        stroke_inside = gr.Checkbox(label='Draw inside the edge', value=False,
                                                    visible=False)
                    stroke_opacity = gr.Slider(0, 100, value=100, step=1,
                                               label='Stroke opacity (%)', visible=False)
                    stroke_color = gr.ColorPicker(label='Stroke colour', value='#000000',
                                                  visible=False)
                    effect_shadow = gr.Checkbox(label='Drop shadow', value=False)
                    shadow_angle = gr.Slider(-180, 180, value=90, step=1,
                                             label='Shadow angle (degrees)', visible=False)
                    shadow_distance = gr.Slider(0, 200, value=20, step=1,
                                                label='Shadow distance (pixels)', visible=False)
                    shadow_blur = gr.Slider(0, 200, value=20, step=1,
                                            label='Shadow blur (pixels)', visible=False)
                    shadow_opacity = gr.Slider(0, 100, value=50, step=1,
                                               label='Shadow opacity (%)', visible=False)
                    shadow_color = gr.ColorPicker(label='Shadow colour', value='#000000', visible=False)
                    effect_overlay = gr.Checkbox(label='Colour overlay', value=False)
                    overlay_opacity = gr.Slider(0, 100, value=100, step=1,
                                                label='Overlay opacity (%)', visible=False)
                    overlay_color = gr.ColorPicker(label='Overlay colour', value='#000000', visible=False)
                    effect_inner_shadow = gr.Checkbox(label='Inner shadow', value=False)
                    inner_shadow_angle = gr.Slider(-180, 180, value=90, step=1,
                                                   label='Inner shadow angle (degrees)', visible=False)
                    inner_shadow_distance = gr.Slider(0, 200, value=10, step=1,
                                                      label='Inner shadow distance (pixels)', visible=False)
                    inner_shadow_blur = gr.Slider(0, 200, value=10, step=1,
                                                  label='Inner shadow blur (pixels)', visible=False)
                    inner_shadow_opacity = gr.Slider(0, 100, value=50, step=1,
                                                     label='Inner shadow opacity (%)', visible=False)
                    inner_shadow_color = gr.ColorPicker(label='Inner shadow colour', value='#000000',
                                                        visible=False)
                    effect_outer_glow = gr.Checkbox(label='Outer glow', value=False)
                    outer_glow_size = gr.Slider(0, 200, value=20, step=1,
                                                label='Outer glow size (pixels)', visible=False)
                    outer_glow_opacity = gr.Slider(0, 100, value=75, step=1,
                                                   label='Outer glow opacity (%)', visible=False)
                    outer_glow_color = gr.ColorPicker(label='Outer glow colour', value='#ffffff',
                                                      visible=False)
                    effect_inner_glow = gr.Checkbox(label='Inner glow', value=False)
                    inner_glow_size = gr.Slider(0, 200, value=10, step=1,
                                                label='Inner glow size (pixels)', visible=False)
                    inner_glow_opacity = gr.Slider(0, 100, value=75, step=1,
                                                   label='Inner glow opacity (%)', visible=False)
                    inner_glow_color = gr.ColorPicker(label='Inner glow colour', value='#ffffff',
                                                      visible=False)
                    apply_effects = gr.Button('Apply layer effects')
                    clear_effects = gr.Button('Clear layer effects')
                    gr.Markdown('Effects need a plain raster layer. Clear removes every effect; '
                                'each apply is one undo step.')
                with gr.Row():
                    up = gr.Button('Move up')
                    down = gr.Button('Move down')
                    remove = gr.Button('Remove layer')
                add_photo = gr.File(label='Photo to add as layer', file_types=['image'], type='filepath')
                add = gr.Button('Add photo layer')
                with gr.Row():
                    undo = gr.Button('Undo', interactive=False, elem_id='editor-undo')
                    redo = gr.Button('Redo', interactive=False, elem_id='editor-redo')
                from retouch.recipes import RECIPES
                recipe = gr.Dropdown(label='Retouch recipe', choices=sorted(RECIPES), value='natural')
                strength = gr.Slider(1, 100, value=100, step=1, label='Result layer strength')
                recipe_mode = gr.Radio(['New layer', 'Regenerate selected recipe layer'],
                                       value='New layer', label='Recipe render target')
                recipe_override = gr.Checkbox(label='Override smoothing and whitening', value=False)
                recipe_smooth = gr.Slider(0, 100, value=30, step=1, label='Recipe skin smoothing', interactive=False)
                recipe_whiten = gr.Slider(0, 100, value=0, step=1, label='Recipe skin whitening', interactive=False)
                extra_recipe_inputs = []
                with gr.Accordion('Eyes, blush, shadow fill and skin detail', open=False):
                    gr.Markdown('Enable only the settings you want to override. Disabled settings '
                                'use the chosen recipe. Skin detail retention runs from 0 to 1.')
                    for control_name, label in EXTRA_RECIPE_CONTROLS.items():
                        spec = RECIPE_OVERRIDE_SPECS[control_name]
                        enabled = gr.Checkbox(label='Override ' + label.lower(), value=False)
                        slider = gr.Slider(spec.min_val, spec.max_val, value=spec.default,
                                           step=.01 if spec.max_val == 1 else 1,
                                           label=label, interactive=False)
                        enabled.change(lambda on: gr.update(interactive=on), enabled, slider,
                                       queue=False)
                        extra_recipe_inputs.extend([enabled, slider])
                load_recipe = gr.Button('Load selected recipe settings')
                gr.Markdown('New layers keep the original composite for later regeneration. '
                            'Regeneration uses that saved source, preserving the selected layer’s '
                            'mask, opacity, placement and name. Later edits underneath are not '
                            'included; choose New layer to capture a fresh composite.')
                render = gr.Button('Render recipe')
                cancel = gr.Button('Cancel recipe render', interactive=False)
                render_status = gr.Markdown('Recipe renderer ready.')
                save_path = gr.Textbox(label='Save project folder', placeholder='/path/to/photo.comp')
                save = gr.Button('Save project')
                export_format = gr.Dropdown(
                    label='Export format', value='PNG (8-bit sRGB)',
                    choices=['PNG (8-bit sRGB)', 'TIFF (16-bit sRGB)'])
                export = gr.Button('Export flattened image')
                export_jpeg_button = gr.Button('Export JPEG')
                jpeg_quality = gr.Slider(
                    label='JPEG quality', minimum=1, maximum=100, step=1, value=85)
                jpeg_background = gr.ColorPicker(label='JPEG background', value='#ffffff')
                download = gr.File(label='Export download', interactive=False)
        with gr.Accordion('Local healing, clone, dodge & burn', open=False):
            local_capture = gr.State(None)
            gr.Markdown('Load the visible composite, then paint the area to change. '
                        'Apply creates a separate masked layer at native resolution. '
                        'Local strokes are drafts until applied; saving keeps applied layers. '
                        'If the document changes, reload and repaint.')
            discard_local = gr.Checkbox(label='Discard local strokes when loading', value=False)
            load_local = gr.Button('Load composite for local retouch')
            local_canvas = gr.ImageEditor(
                label='Paint local retouch region', type='numpy', image_mode='RGBA',
                sources=(), transforms=(), format='png', height=500,
                brush=gr.Brush(colors=['#ff3366'], color_mode='fixed', default_color='#ff3366'))
            with gr.Row():
                local_operation = gr.Radio(['Heal', 'Clone', 'Dodge', 'Burn'], value='Heal', label='Local operation')
                heal_method = gr.Radio(['Telea', 'Navier-Stokes'], value='Telea', label='Healing method')
            with gr.Column(visible=False) as clone_controls:
                gr.Markdown('Paint the destination. Each painted pixel copies from the visible '
                            'composite at that pixel plus this source offset, in native image pixels. '
                            'Negative X samples left; negative Y samples above. For example, X = -50 '
                            'copies texture from 50 pixels left. Out-of-canvas sources are refused.')
                clone_x = gr.Number(value=0, precision=0, label='Clone source offset X (pixels)')
                clone_y = gr.Number(value=0, precision=0, label='Clone source offset Y (pixels)')
                show_clone_picker = gr.Button('Load clone source picker')
                clone_placement = gr.State(None)
                clone_target = gr.Radio(['Source', 'Destination'], value='Source', label='Point to pick')
                clone_picker_image = gr.Image(label='Click source, then destination', type='numpy',
                                              interactive=False, sources=(), format='png', height=400,
                                              show_download_button=False)
                clone_picker_status = gr.Markdown('Load the local composite, then load this picker. '
                                                  'Green marks the source; pink marks the destination. '
                                                  'Paint on the local retouch canvas above.')
            local_opacity = gr.Slider(1, 100, value=100, step=1, label='Local result opacity (%)')
            limit_local = gr.Checkbox(label='Limit local strokes to the current selection', value=False)
            apply_local = gr.Button('Apply local retouch as new layer')
            local_status = gr.Markdown('Load a composite to start local retouch.')
        status = gr.Markdown('Open a photo or project to start.')
        outputs = [session, preview, layers, name, visible, opacity, mask_enabled,
                   canvas, undo, redo, status, blend, position_x, position_y, flip_x, flip_y,
                   layer_width, layer_height, rotation, parent_folder, clipping_source, group_members,
                   selection_preview, grade_kind, grade_exposure, grade_offset, grade_gamma, grade_black,
                   grade_white, grade_red, grade_green, grade_blue, grade_saturation, grade_vibrance,
                   grade_temperature, grade_tint, grade_brightness, grade_contrast, grade_hue,
                   effect_stroke, stroke_size, stroke_inside, stroke_opacity, stroke_color,
                   effect_shadow, shadow_angle, shadow_distance, shadow_blur, shadow_opacity,
                   shadow_color, effect_overlay, overlay_opacity, overlay_color,
                   effect_inner_shadow, inner_shadow_angle, inner_shadow_distance,
                   inner_shadow_blur, inner_shadow_opacity, inner_shadow_color,
                   effect_outer_glow, outer_glow_size, outer_glow_opacity, outer_glow_color,
                   effect_inner_glow, inner_glow_size, inner_glow_opacity, inner_glow_color,
                   *curve_inputs, grain_amount, grain_size, grain_roughness, texture_seed,
                   noise_amount, noise_gaussian, noise_monochromatic, curves_plot]

        def loading_local(current, strokes, local_strokes, discard):
            from .local_retouch import capture_composite
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase the layer-mask strokes first.')
            if has_strokes(local_strokes) and not discard:
                raise gr.Error('Apply local retouch or choose Discard local strokes when loading.')
            try:
                brush, captured = capture_composite(current.history)
            except (ValueError, OSError, cv2.error) as exc:
                raise gr.Error(str(exc)) from exc
            return brush, captured, 'Composite loaded. Paint the region to heal, clone, dodge or burn.', False

        load_local.click(loading_local, [session, canvas, local_canvas, discard_local],
                         [local_canvas, local_capture, local_status, discard_local],
                         concurrency_id='layer-editor')

        def applying_local(current, captured, local_strokes, strokes, operation,
                           amount, method, limit, source_x=0, source_y=0):
            from .local_retouch import apply_local_retouch, capture_composite
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase the layer-mask strokes first.')
            if limit and current.selection is None:
                raise gr.Error('Create a selection first, or turn off the selection limit.')
            try:
                layer_id = apply_local_retouch(
                    current.history, captured, local_strokes, operation, amount, method,
                    selection=current.selection if limit else None,
                    clone_offset=(source_x, source_y))
                current.selected_id = layer_id
                brush, captured = capture_composite(current.history)
            except (ValueError, OSError, cv2.error) as exc:
                raise gr.Error(str(exc)) from exc
            return (*_view(current), brush, captured, operation + ' added as a masked layer. Undo restores the previous image.')

        apply_local.click(applying_local,
                          [session, local_capture, local_canvas, canvas, local_operation,
                           local_opacity, heal_method, limit_local, clone_x, clone_y],
                          outputs + [local_canvas, local_capture, local_status], concurrency_id='layer-editor')
        local_operation.change(
            lambda value: (gr.update(visible=value == 'Heal'),
                           gr.update(value=100 if value in ('Heal', 'Clone') else 25),
                           gr.update(visible=value == 'Clone')),
            local_operation, [heal_method, local_opacity, clone_controls], queue=False)

        def loading_clone_picker(current, captured):
            from .clone_picker import prepare_clone_picker
            if current is None:
                raise gr.Error('Open a document first')
            try:
                placement = prepare_clone_picker(current.history, captured)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return (placement.preview, placement, 0, 0, 'Click the source texture, then its destination.',
                    'Source')

        def picking_clone_point(current, placement, target, evt: gr.SelectData):
            from .clone_picker import pick_clone_point
            if current is None:
                raise gr.Error('Open a document first')
            try:
                placement, image, offset, message = pick_clone_point(
                    current.history, placement, evt.index, target)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return (image, placement, *offset, message,
                    'Destination' if target == 'Source' else target)

        clone_picker_outputs = [clone_picker_image, clone_placement, clone_x, clone_y,
                                clone_picker_status, clone_target]
        show_clone_picker.click(loading_clone_picker, [session, local_capture],
                                clone_picker_outputs, concurrency_id='layer-editor')
        clone_picker_image.select(picking_clone_point, [session, clone_placement, clone_target],
                                  clone_picker_outputs, concurrency_id='layer-editor')

        def recovery_choices():
            entries = list_recoveries()
            return gr.update(choices=[(e.label, e.entry_id) for e in entries],
                             value=entries[0].entry_id if entries else None)

        def opening(current, photo_path, path, discard_edits, strokes, kind):
            if current and current.job and not current.job.finalized:
                raise gr.Error('Cancel or finish the recipe render before opening another document.')
            if current and (current.history.dirty or has_strokes(strokes)) and not discard_edits:
                raise gr.Error('Save or select “Discard unsaved edits” before opening another document.')
            recovery_id = None
            try:
                if kind == 'photo':
                    if not photo_path:
                        raise ValueError('Choose a photo first')
                    pixels = _photo(photo_path)
                    doc = Document(pixels.shape[1], pixels.shape[0])
                    title = Path(photo_path).stem
                    doc.add_image(pixels, title)
                elif kind == 'recovery':
                    # ``path`` carries the chosen entry id.
                    entry = next((e for e in list_recoveries() if e.entry_id == path), None)
                    if entry is None:
                        raise ValueError('That autosave is no longer available; refresh the list')
                    doc, title, recovery_id = load_recovery(path), entry.title, path
                else:
                    doc = load_project(path)
                    title = Path(path).name
                # Check that this subset can be rendered before replacing the session.
                preview_document(doc)
                # A restored document stays unsaved until the user saves it.
                replacement = EditorSession.create(doc, saved=kind == 'project', title=title,
                                                   recovery_id=recovery_id)
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc
            if current:
                current.close()
            message = ('Restored autosaved work. Save the project to keep it.'
                       if kind == 'recovery' else '')
            return (*_view(replacement, message), False)

        for button, kind in ((open_photo, 'photo'), (open_project, 'project')):
            def load(current, p, path, d, strokes, kind=kind):
                return opening(current, p, path, d, strokes, kind)
            button.click(load, [session, photo, project_path, discard, canvas],
                         outputs + [discard], concurrency_id='layer-editor')

        def restoring(current, entry_id, d, strokes):
            if not entry_id:
                raise gr.Error('Choose an autosaved document first')
            return (*opening(current, None, entry_id, d, strokes, 'recovery'), recovery_choices())
        restore.click(restoring, [session, recoveries, discard, canvas],
                      outputs + [discard, recoveries], concurrency_id='layer-editor')

        def discarding_recovery(entry_id):
            if not entry_id:
                raise gr.Error('Choose an autosaved document first')
            try:
                discard_recovery(entry_id)
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc
            return recovery_choices()
        discard_saved.click(discarding_recovery, recoveries, recoveries, concurrency_id='layer-editor')
        refresh_recoveries.click(recovery_choices, None, recoveries, queue=False)

        def offering_recoveries():
            update = recovery_choices()
            return update, gr.update(open=bool(update['choices']))
        tab.select(offering_recoveries, None, [recoveries, recovery_panel], queue=False)

        def action(current, selected, strokes, operation, *args):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes) and operation not in ('mask', 'heal', 'clone'):
                raise gr.Error('Apply or erase the current brush strokes first.')
            previous_selection = current.selected_id
            try:
                if operation == 'select':
                    current.history.document.layer(selected)
                    current.selected_id = selected
                else:
                    layer = current.selected()
                    if operation in ('properties', 'transform', 'resize', 'fit', 'up', 'down', 'remove', 'mask', 'duplicate', 'invert_mask', 'reset_mask', 'links', 'ungroup', 'selection_mask', 'update_grade', 'apply_effects', 'clear_effects') and layer is None:
                        raise ValueError('Select a layer first')
                    if operation == 'properties':
                        chosen_blend = args[4] if len(args) > 4 else layer.blend_mode
                        if args[1] and chosen_blend not in RENDER_BLEND_MODES:
                            raise ValueError('Choose a supported blend mode before revealing this layer')
                        current.history.update_layer(layer.id, name=args[0], visible=args[1],
                                                     opacity=args[2] / 100, mask_enabled=args[3],
                                                     blend_mode=chosen_blend)
                    elif operation == 'transform':
                        for value in args[:2]:
                            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                                    or not math.isfinite(value)):
                                raise ValueError('Layer positions must be finite numbers')
                        current.history.update_layer(layer.id,
                                                     origin=(float(args[0]), float(args[1])),
                                                     flip_x=args[2], flip_y=args[3],
                                                     rotation=args[4] if len(args) > 4 else layer.rotation)
                    elif operation == 'links':
                        current.history.update_layer(layer.id, parent_id=args[0] or None,
                                                     mask_source_id=args[1] or None)
                    elif operation in ('apply_effects', 'clear_effects'):
                        from .document import (LayerEffects, StrokeEffect, ShadowEffect,
                                               ColorOverlayEffect, InnerShadowEffect,
                                               OuterGlowEffect, InnerGlowEffect)
                        if operation == 'clear_effects':
                            effects = None
                        else:
                            stroke = None
                            if args[0]:
                                red, green, blue = _rgb(args[4])
                                stroke = StrokeEffect(size=args[1], inside=args[2],
                                                      red=red, green=green,
                                                      blue=blue, opacity=args[3] / 100)
                            shadow = None
                            if args[5]:
                                red, green, blue = _rgb(args[10])
                                shadow = ShadowEffect(angle=args[6], distance=args[7],
                                                      blur=args[8], red=red, green=green,
                                                      blue=blue, opacity=args[9] / 100)
                            overlay = None
                            if args[11]:
                                red, green, blue = _rgb(args[13])
                                overlay = ColorOverlayEffect(red=red, green=green, blue=blue,
                                                             opacity=args[12] / 100)
                            inner_shadow = None
                            if args[14]:
                                red, green, blue = _rgb(args[19])
                                inner_shadow = InnerShadowEffect(angle=args[15], distance=args[16],
                                                                 blur=args[17], red=red, green=green,
                                                                 blue=blue, opacity=args[18] / 100)
                            outer_glow = None
                            if args[20]:
                                red, green, blue = _rgb(args[23])
                                outer_glow = OuterGlowEffect(size=args[21], red=red,
                                                             green=green, blue=blue,
                                                             opacity=args[22] / 100)
                            inner_glow = None
                            if args[24]:
                                red, green, blue = _rgb(args[27])
                                inner_glow = InnerGlowEffect(size=args[25], red=red,
                                                             green=green, blue=blue,
                                                             opacity=args[26] / 100)
                            effects = LayerEffects(stroke=stroke, shadow=shadow,
                                                   colorOverlay=overlay, innerShadow=inner_shadow,
                                                   outerGlow=outer_glow,
                                                   innerGlow=inner_glow) if (
                                stroke or shadow or overlay or inner_shadow
                                or outer_glow or inner_glow) else None
                        if (effects is not None and (layer.is_group or layer.adjustment is not None
                                                     or layer.mask_source_id is not None)):
                            raise ValueError('Layer effects need a plain raster layer')
                        current.history.update_layer(layer.id, effects=effects)
                    elif operation == 'group':
                        current.history.group_layers(args[0] or [])
                        current.selected_id = current.history.document.active_layer_id
                    elif operation == 'ungroup':
                        current.history.ungroup_layer(layer.id)
                        current.selected_id = current.history.document.active_layer_id
                    elif operation == 'selection':
                        from .selections import selection, combine
                        doc = current.history.document
                        new = selection(doc.width, doc.height, *args[:7])
                        old = current.selection if current.selection is not None else np.zeros(new.shape, np.uint8)
                        current.selection = combine(old, new, args[7])
                    elif operation == 'wand':
                        from .selections import combine, wand_selection
                        doc = current.history.document
                        source = composite(doc)
                        radius = {'Point': 0, '3x3': 1, '5x5': 2}[args[3]] \
                            if isinstance(args[3], str) else int(args[3])
                        new, count = wand_selection(source, args[0], args[1],
                                                    tolerance=args[2], radius=radius,
                                                    contiguous=args[4])
                        old = current.selection if current.selection is not None \
                            else np.zeros(new.shape, np.uint8)
                        merged = combine(old, new, args[5])
                        # An empty Replace deselects, matching invert_selection.
                        current.selection = None if (not merged.any()
                                                     and args[5] == 'Replace') else merged
                    elif operation == 'automatic_selection':
                        from .semantic_selection import automatic_selection
                        from .selections import combine
                        doc = current.history.document
                        captured = (doc.document_id, current.history.revision)
                        new = automatic_selection(composite(doc), args[0], args[1], current.workspace)
                        if captured != (current.history.document.document_id, current.history.revision):
                            raise ValueError('Document changed; create the automatic selection again.')
                        old = current.selection if current.selection is not None else np.zeros(new.shape, np.uint8)
                        merged = combine(old, new, args[2])
                        current.selection = merged if merged.any() else None
                    elif operation == 'clear_selection':
                        current.selection = None
                    elif operation == 'invert_selection':
                        # Upstream invertSelection: no selection is a no-op,
                        # and the inverse of everything deselects entirely
                        # (Selection.swift:354-361) rather than leaving an
                        # invisible empty mask.
                        if current.selection is not None:
                            inverse = 255 - current.selection
                            current.selection = inverse if inverse.any() else None
                    elif operation in ('expand_selection', 'contract_selection', 'border_selection'):
                        if current.selection is None:
                            raise ValueError('Create a selection first')
                        from .selections import border as selection_border
                        from .selections import contract as selection_contract
                        from .selections import expand as selection_expand
                        ops = {'expand_selection': selection_expand,
                               'contract_selection': selection_contract,
                               'border_selection': selection_border}
                        current.selection = ops[operation](current.selection, args[0])
                    elif operation == 'move_selection':
                        if current.selection is None:
                            raise ValueError('Create a selection first')
                        from .selections import move as selection_move
                        current.selection = selection_move(current.selection, args[0], args[1])
                    elif operation == 'selection_mask':
                        if current.selection is None:
                            raise ValueError('Create a selection first')
                        current.history.selection_mask(layer.id, current.selection, args[0])
                    elif operation in ('add_grade', 'update_grade'):
                        control_args = ((*args[:16], list(args[16:20]), *args[20:])
                                        if len(args) > 16 else args)
                        settings = controls_adjustment(
                            *control_args, existing=layer.adjustment
                            if operation == 'update_grade' and layer else None)
                        if operation == 'update_grade':
                            if layer.adjustment is None:
                                raise ValueError('Select an adjustment layer first')
                            current.history.update_layer(layer.id, adjustment=settings)
                        else:
                            doc = current.history.document
                            new = Layer(args[0], size=(doc.width, doc.height), adjustment=settings,
                                        parent_id=(layer.id if layer.is_group else layer.parent_id) if layer else None)
                            current.history.add_layer(new, doc.index_of(layer.id) + 1 if layer else None)
                            current.selected_id = new.id
                    elif operation in ('crop', 'crop_selection'):
                        from .selections import selection_bbox
                        if operation == 'crop':
                            current.history.crop_document(args[0], args[1], args[2], args[3])
                        else:
                            if current.selection is None:
                                raise ValueError('Create a selection first')
                            box = selection_bbox(current.selection)
                            if box is None:
                                raise ValueError('Create a selection first')
                            current.history.crop_document(*box)
                        current.selection = None
                    elif operation == 'resize':
                        current.history.resize_layer(layer.id, args[0], args[1],
                                                     keep_aspect=args[2] if len(args) > 2 else False,
                                                     axis=args[3] if len(args) > 3 else 'Width')
                    elif operation == 'fit':
                        current.history.fit_layer(layer.id)
                    elif operation in ('up', 'down'):
                        index = current.history.document.index_of(layer.id)
                        current.history.move_layer(layer.id, index + (1 if operation == 'up' else -1))
                    elif operation == 'remove':
                        current.history.remove_layer(layer.id)
                    elif operation == 'duplicate':
                        current.history.duplicate_layer(layer.id)
                        current.selected_id = current.history.document.active_layer_id
                    elif operation == 'invert_mask':
                        current.history.invert_mask(layer.id)
                    elif operation == 'reset_mask':
                        current.history.reset_mask(layer.id)
                    elif operation == 'mask':
                        current.apply_mask(strokes, args[0] if args else None)
                    elif operation == 'heal':
                        current.paint_pixels(strokes, 'heal', heal_mode=args[0], opacity=args[1])
                    elif operation == 'clone':
                        current.paint_pixels(strokes, 'clone', opacity=args[0],
                                             offset=(args[1], args[2]),
                                             softness=args[3] if len(args) > 3 else 0)
                    elif operation == 'add':
                        if not args[0]:
                            raise ValueError('Choose a photo to add first')
                        pixels = _photo(args[0])
                        new = Layer(Path(args[0]).stem, pixels=pixels)
                        current.history.add_layer(new)
                        current.selected_id = new.id
                    elif operation == 'undo':
                        current.history.undo()
                    elif operation == 'redo':
                        current.history.redo()
                return _view(current)
            except (ValueError, OSError, KeyError) as exc:
                current.selected_id = previous_selection
                raise gr.Error(str(exc)) from exc

        def select(current, selected, strokes):
            return action(current, selected, strokes, 'select')
        layers.input(select, [session, layers, canvas], outputs, concurrency_id='layer-editor')
        for button, operation in ((properties, 'properties'), (up, 'up'), (down, 'down'),
                                  (remove, 'remove'), (apply_mask, 'mask'), (heal, 'heal'), (clone, 'clone'), (add, 'add'),
                                  (undo, 'undo'), (redo, 'redo'), (transform, 'transform'), (resize, 'resize'),
                                  (duplicate, 'duplicate'), (invert_mask, 'invert_mask'),
                                  (reset_mask, 'reset_mask'), (fit, 'fit'), (links, 'links'), (group, 'group'),
                                  (ungroup, 'ungroup'), (create_selection, 'selection'),
                                  (wand_button, 'wand'), (auto_select, 'automatic_selection'),
                                  (crop_canvas, 'crop'), (crop_selection, 'crop_selection'),
                                  (clear_selection, 'clear_selection'), (invert_selection, 'invert_selection'),
                                  (expand_selection, 'expand_selection'),
                                  (contract_selection, 'contract_selection'),
                                  (border_selection, 'border_selection'),
                                  (move_selection, 'move_selection'),
                                  (mask_selection, 'selection_mask'), (add_grade, 'add_grade'),
                                  (update_grade, 'update_grade'),
                                  (apply_effects, 'apply_effects'), (clear_effects, 'clear_effects')):
            def edit(current, selected, strokes, *args, operation=operation):
                return action(current, selected, strokes, operation, *args)
            extras = {
                'properties': [name, visible, opacity, mask_enabled, blend],
                'transform': [position_x, position_y, flip_x, flip_y, rotation],
                'resize': [layer_width, layer_height, keep_aspect, resize_axis],
                'add': [add_photo], 'links': [parent_folder, clipping_source],
                'crop': [crop_x, crop_y, crop_width, crop_height],
                'mask': [mask_action],
                'heal': [heal_mode, heal_opacity],
                'clone': [heal_opacity, clone_dx, clone_dy, clone_softness],
                'group': [group_members],
                'selection': [selection_kind, selection_x, selection_y, selection_width,
                              selection_height, polygon, feather, selection_mode],
                'selection_mask': [mask_mode],
                'automatic_selection': [semantic_region, feather, selection_mode],
                'wand': [wand_x, wand_y, wand_tolerance, wand_sample,
                         wand_contiguous, selection_mode],
                'expand_selection': [selection_amount],
                'contract_selection': [selection_amount],
                'border_selection': [selection_amount],
                'move_selection': [move_dx, move_dy],
                'add_grade': [grade_kind, grade_exposure, grade_offset, grade_gamma,
                              grade_black, grade_white, grade_red, grade_green, grade_blue,
                              grade_saturation, grade_vibrance, grade_temperature, grade_tint,
                              grade_brightness, grade_contrast, grade_hue,
                              *curve_inputs, grain_amount, grain_size, grain_roughness, texture_seed,
                              noise_amount, noise_gaussian, noise_monochromatic],
                'update_grade': [grade_kind, grade_exposure, grade_offset, grade_gamma,
                                 grade_black, grade_white, grade_red, grade_green, grade_blue,
                                 grade_saturation, grade_vibrance, grade_temperature, grade_tint,
                                 grade_brightness, grade_contrast, grade_hue,
                                 *curve_inputs, grain_amount, grain_size, grain_roughness, texture_seed,
                                 noise_amount, noise_gaussian, noise_monochromatic],
                'apply_effects': [effect_stroke, stroke_size, stroke_inside, stroke_opacity,
                                  stroke_color, effect_shadow, shadow_angle, shadow_distance,
                                  shadow_blur, shadow_opacity, shadow_color, effect_overlay,
                                  overlay_opacity, overlay_color,
                                  effect_inner_shadow, inner_shadow_angle, inner_shadow_distance,
                                  inner_shadow_blur, inner_shadow_opacity, inner_shadow_color,
                                  effect_outer_glow, outer_glow_size, outer_glow_opacity,
                                  outer_glow_color,
                                  effect_inner_glow, inner_glow_size, inner_glow_opacity,
                                  inner_glow_color],
                'clear_effects': [],
            }
            extra = extras.get(operation, [])
            button.click(edit, [session, layers, canvas] + extra, outputs, concurrency_id='layer-editor')

        grade_kind.change(
            lambda kind: [gr.update(visible=kind == 'Curves'),
                          gr.update(visible=kind == 'Grain'),
                          gr.update(visible=kind == 'Add Noise'),
                          gr.update(visible=kind in ('Grain', 'Add Noise'))],
            grade_kind, [curves_controls, grain_controls, noise_controls, seed_controls], queue=False)
        for table in curve_inputs:
            table.input(curve_graph, curve_inputs, curves_plot, queue=False)

        def reset_curve_controls():
            tables = curve_tables({})
            return (*tables, curve_graph(*tables))

        reset_curves.click(reset_curve_controls, None, curve_inputs + [curves_plot], queue=False)

        # Show a shadow's or overlay's settings only while its effect is on.
        effect_stroke.change(
            lambda on: [gr.update(visible=on)] * 4, effect_stroke,
            [stroke_size, stroke_inside, stroke_opacity, stroke_color], queue=False)
        effect_shadow.change(
            lambda on: [gr.update(visible=on)] * 5, effect_shadow,
            [shadow_angle, shadow_distance, shadow_blur, shadow_opacity, shadow_color],
            queue=False)
        effect_overlay.change(
            lambda on: [gr.update(visible=on)] * 2, effect_overlay,
            [overlay_opacity, overlay_color], queue=False)
        effect_inner_shadow.change(
            lambda on: [gr.update(visible=on)] * 5, effect_inner_shadow,
            [inner_shadow_angle, inner_shadow_distance, inner_shadow_blur,
             inner_shadow_opacity, inner_shadow_color], queue=False)
        effect_outer_glow.change(
            lambda on: [gr.update(visible=on)] * 3, effect_outer_glow,
            [outer_glow_size, outer_glow_opacity, outer_glow_color], queue=False)
        effect_inner_glow.change(
            lambda on: [gr.update(visible=on)] * 3, effect_inner_glow,
            [inner_glow_size, inner_glow_opacity, inner_glow_color], queue=False)

        def grade_visibility(kind):
            levels = kind == 'Levels'
            exposure = kind == 'Exposure'
            return [
                gr.update(visible=exposure),  # exposure
                gr.update(visible=exposure),  # offset
                gr.update(visible=exposure or levels),  # gamma
                *([gr.update(visible=levels)] * 5),  # black, white, RGB gamma
                gr.update(visible=kind == 'Saturation'),
                gr.update(visible=kind == 'Vibrance'),
                gr.update(visible=kind == 'Temperature'),
                gr.update(visible=kind == 'Tint'),
                gr.update(visible=kind == 'Brightness'),
                gr.update(visible=kind == 'Contrast'),
                gr.update(visible=kind == 'Hue'),
            ]

        grade_kind.change(
            grade_visibility, grade_kind,
            [grade_exposure, grade_offset, grade_gamma, grade_black, grade_white,
             grade_red, grade_green, grade_blue, grade_saturation, grade_vibrance,
             grade_temperature, grade_tint, grade_brightness, grade_contrast, grade_hue],
            queue=False)

        def saving(current, path, strokes):
            if current is None or not path:
                raise gr.Error('Open a document and enter a save path first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase brush strokes before saving')
            try:
                current.history.save(Path(path).expanduser())
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc
            return _view(current, 'Project saved.')
        save.click(saving, [session, save_path, canvas], outputs, concurrency_id='layer-editor')

        def exporting(current, strokes, chosen_format):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase brush strokes before exporting')
            bit_depth, suffix = (16, '.tif') if '16-bit' in chosen_format else (8, '.png')
            path = Path(current.workspace) / (uuid.uuid4().hex + suffix)
            export_png(current.history.document, path, bit_depth=bit_depth)
            return str(path)
        export.click(exporting, [session, canvas, export_format], download, concurrency_id='layer-editor')

        def exporting_jpeg(current, strokes, quality, background):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase brush strokes before exporting')
            if not background:
                background = '#ffffff'
            channels = tuple(int(background.lstrip('#')[i:i + 2], 16) / 255.0
                             for i in (0, 2, 4))
            path = Path(current.workspace) / (uuid.uuid4().hex + '.jpg')
            export_jpeg(current.history.document, path,
                        quality=float(quality) / 100.0, background=channels)
            return str(path)
        export_jpeg_button.click(
            exporting_jpeg, [session, canvas, jpeg_quality, jpeg_background],
            download, concurrency_id='layer-editor')

        recipe_mode.change(lambda mode: gr.update(interactive=mode == 'New layer'),
                           recipe_mode, strength, queue=False)
        recipe_override.change(lambda enabled: [gr.update(interactive=enabled)] * 2,
                               recipe_override, [recipe_smooth, recipe_whiten], queue=False)

        from .retouch_jobs import prepare_job, run_job, attach_result

        def prepare_recipe(current, chosen_recipe, chosen_strength, strokes,
                           target_mode='New layer', use_overrides=False, smooth=30, whiten=0,
                           *extra_values):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase brush strokes before rendering')
            if current.job and not current.job.finalized:
                raise gr.Error('A recipe render is already active')
            try:
                if target_mode not in ('New layer', 'Regenerate selected recipe layer'):
                    raise ValueError('Choose a recipe render target')
                target = current.selected_id if target_mode != 'New layer' else None
                if target_mode != 'New layer' and target is None:
                    raise ValueError('Select a recipe layer first')
                overrides = dict(smooth=smooth, whiten=whiten) if use_overrides else {}
                if extra_values and len(extra_values) != len(extra_recipe_inputs):
                    raise ValueError('Incomplete recipe controls; reload the editor')
                for name, enabled, value in zip(EXTRA_RECIPE_CONTROLS,
                                                extra_values[::2], extra_values[1::2]):
                    if enabled:
                        overrides[name] = value
                current.job = prepare_job(current.history, current.workspace, chosen_recipe,
                                          chosen_strength, target_layer_id=target, overrides=overrides)
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc
            return (current, 'Recipe and document revision captured.',
                    gr.update(interactive=False), gr.update(interactive=True))

        def rendering(current):
            if current is None or current.job is None:
                return
            try:
                yield from run_job(current.job)
            except (ValueError, OSError) as exc:
                current.job.status, current.job.finished = 'failed', True
                yield 'Recipe render failed: ' + str(exc)

        def finish_recipe(current, strokes):
            if current is None or current.job is None:
                raise gr.Error('Recipe session is unavailable')
            message = 'Recipe render ' + current.job.status + '.'
            if current.job.status == 'completed':
                try:
                    if has_strokes(strokes):
                        current.job.status = 'stale'
                        raise ValueError('Unapplied strokes changed the canvas; apply them and rerun the recipe')
                    current.selected_id = attach_result(current.history, current.job)
                    message = ('Recipe layer regenerated; its mask and properties were preserved.'
                               if current.job.target_layer_id else 'Recipe result added as a new masked layer.')
                except (ValueError, OSError) as exc:
                    message = str(exc)
            # Keep unapplied strokes on rejection; never silently clear the canvas.
            view = list(_view(current, message))
            if has_strokes(strokes):
                view[7] = gr.update()
            current.job.finalized = True
            return (*view, message, gr.update(interactive=True), gr.update(interactive=False))

        def loading_recipe_settings(current):
            if current is None or current.selected() is None or current.selected().recipe_settings is None:
                raise gr.Error('Select a recipe layer with saved settings first')
            layer = current.selected()
            settings = layer.recipe_settings
            overrides = settings['overrides']
            extras = []
            for name in EXTRA_RECIPE_CONTROLS:
                enabled = name in overrides
                extras.extend([enabled, gr.update(value=overrides.get(name, RECIPE_OVERRIDE_SPECS[name].default),
                                                  interactive=enabled)])
            skin_enabled = 'smooth' in overrides or 'whiten' in overrides
            return (settings['recipe'], layer.opacity * 100, skin_enabled,
                    gr.update(value=overrides.get('smooth', 30), interactive=skin_enabled),
                    gr.update(value=overrides.get('whiten', 0), interactive=skin_enabled),
                    'Regenerate selected recipe layer', 'Saved recipe settings loaded.', *extras)

        load_recipe.click(loading_recipe_settings, session,
                          [recipe, strength, recipe_override, recipe_smooth, recipe_whiten,
                           recipe_mode, render_status] + extra_recipe_inputs, concurrency_id='layer-editor')

        def cancel_recipe(current):
            if current is None or current.job is None:
                return 'No active recipe render.'
            current.job.cancel()
            return 'Cancellation requested; no cancelled result will be attached.'

        render.click(prepare_recipe, [session, recipe, strength, canvas, recipe_mode,
                                      recipe_override, recipe_smooth, recipe_whiten] + extra_recipe_inputs,
                     [session, render_status, render, cancel], concurrency_id='layer-editor').success(
            rendering, session, render_status, concurrency_id='layer-editor-recipe').then(
            finish_recipe, [session, canvas], outputs + [render_status, render, cancel],
            concurrency_id='layer-editor')
        cancel.click(cancel_recipe, session, render_status, queue=False)
        status.change(None, status, None, js="""(text) => {
            window.retouchLayerDirty = text.includes('Unsaved changes');
            window.retouchLayerStrokeDirty = text.includes('Unapplied strokes');
            if (!window.retouchLayerUnloadGuard) {
                window.retouchLayerUnloadGuard = true;
                window.addEventListener('beforeunload', (e) => {
                    if (window.retouchLayerDirty || window.retouchLayerStrokeDirty) {
                        e.preventDefault(); e.returnValue = '';
                    }
                });
            }
        }""")
        canvas.input(None, canvas, None, js="""(value) => {
            window.retouchLayerStrokeDirty = !!(value && value.layers && value.layers.length);
        }""")
