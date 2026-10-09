"""Session-owned layer editor in the existing Gradio desktop shell."""
import shutil
import io
import math
import tempfile
import uuid
import threading
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageCms, ImageOps

from .commands import DocumentHistory
from .composite import RENDER_BLEND_MODES, composite, export_png
from .document import Document, Layer, MAX_SURFACE_PIXELS
from .project_store import load_project

PREVIEW_SIDE = 1024


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
        layer.origin = tuple(round(v * scale) for v in original.origin)
        layer.size = tuple(max(1, round(v * scale)) for v in original.size)
        if original.pixels is not None:
            # Pillow interpolates RGBA in premultiplied form.
            layer.pixels = np.asarray(Image.fromarray(original.pixels).resize(
                layer.size, Image.Resampling.LANCZOS))
        if original.mask is not None:
            layer.mask = cv2.resize(original.mask, layer.size,
                                    interpolation=cv2.INTER_LINEAR)
        preview.layers.append(layer)
    return composite(preview)


def has_strokes(canvas):
    return bool(canvas and any(np.asarray(layer)[..., 3].any()
                              for layer in canvas.get('layers', [])))


@dataclass
class EditorSession:
    history: DocumentHistory
    selected_id: str
    workspace: str
    job: object = None

    @classmethod
    def create(cls, document, saved=False):
        return cls(DocumentHistory(document, saved=saved),
                   document.active_layer_id or (document.layers[-1].id if document.layers else None),
                   tempfile.mkdtemp(prefix='retouch-layer-editor-'))

    def close(self):
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

    def apply_mask(self, canvas):
        layer = self.selected()
        if layer is None or layer.pixels is None:
            raise ValueError('Select a pixel layer first')
        width, height = layer.pixels.shape[1], layer.pixels.shape[0]
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
            value = rgba[..., :3].mean(axis=2)
            mask = np.rint(mask * (1 - alpha) + value * alpha).astype(np.uint8)
            painted = True
        if painted:
            self.history.set_mask(layer.id, mask, layer.mask_enabled)
        return painted


def _view(session, message=''):
    if session is None:
        return (None, None, gr.update(choices=[], value=None), '', True, 100,
                True, None, gr.update(interactive=False), gr.update(interactive=False),
                'Open a photo or project to start.', gr.update(choices=list(RENDER_BLEND_MODES), value='Normal'),
                0, 0, False, False, 1, 1)
    doc = session.history.document
    layer = session.selected()
    canvas = None
    if layer is not None and layer.pixels is not None:
        scale = min(1, PREVIEW_SIDE / max(layer.size))
        size = tuple(max(1, round(v * scale)) for v in layer.size)
        image = np.asarray(Image.fromarray(layer.pixels).resize(size))
        if layer.flip_y:
            image = image[::-1]
        if layer.flip_x:
            image = image[:, ::-1]
        canvas = {'background': image, 'layers': [], 'composite': image}
    state = 'Unsaved changes' if session.history.dirty else 'Saved'
    return (session, preview_document(doc),
            gr.update(choices=[(l.name, l.id) for l in reversed(doc.layers)],
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
            layer.size[0] if layer else 1, layer.size[1] if layer else 1)


def build_editor_tab():
    """Add a complete independent tab inside the application's Tabs block."""
    with gr.Tab('Layer Editor'):
        session = gr.State(None, delete_callback=lambda value: value.close() if value else None)
        gr.Markdown('Open a photo or local `.comp` folder. White reveals; black hides. '
                    'Apply brush strokes before changing layers. Export keeps native resolution.')
        with gr.Row():
            with gr.Column():
                photo = gr.File(label='Photo', file_types=['image'], type='filepath', height=100)
                open_photo = gr.Button('Open photo')
            with gr.Column():
                project_path = gr.Textbox(label='Project folder', placeholder='/path/to/photo.comp',
                                          lines=1, max_lines=1)
                open_project = gr.Button('Open project')
        discard = gr.Checkbox(label='Discard unsaved edits and unapplied strokes when opening', value=False)
        with gr.Row():
            with gr.Column(scale=2):
                preview = gr.Image(label='Composite preview', interactive=False, image_mode='RGBA')
                canvas = gr.ImageEditor(label='Paint selected layer mask', type='numpy',
                                        image_mode='RGBA', sources=(), transforms=(),
                                        format='png',
                                        brush=gr.Brush(colors=['#ffffff', '#000000'],
                                                       color_mode='fixed', default_color='#ffffff'),
                                        height=450)
                gr.Markdown('The canvas shows the layer before its mask. '
                            'Paint white to reveal or black to hide; the eraser removes unapplied strokes.')
                apply_mask = gr.Button('Apply mask strokes')
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
                gr.Markdown('Invert mask swaps revealed and hidden areas. '
                            'A layer without a mask becomes fully masked out; '
                            'enable its mask to see the change.')
                with gr.Accordion('Position & flips', open=False):
                    gr.Markdown('Position uses canvas pixels from the top-left. '
                                'Negative positions are allowed; pixels outside the canvas are clipped. '
                                'Flips move the mask with the layer.')
                    with gr.Row():
                        position_x = gr.Number(label='X position', value=0, precision=0)
                        position_y = gr.Number(label='Y position', value=0, precision=0)
                    flip_x = gr.Checkbox(label='Flip horizontally', value=False)
                    flip_y = gr.Checkbox(label='Flip vertically', value=False)
                    transform = gr.Button('Apply position & flips')
                    with gr.Row():
                        layer_width = gr.Number(label='Layer width (pixels)', value=1, precision=0)
                        layer_height = gr.Number(label='Layer height (pixels)', value=1, precision=0)
                    gr.Markdown('Resizing preserves the original pixels. Set both dimensions '
                                'to keep the aspect ratio; changing one stretches the layer.')
                    resize = gr.Button('Apply layer size')
                with gr.Row():
                    up = gr.Button('Move up')
                    down = gr.Button('Move down')
                    remove = gr.Button('Remove layer')
                add_photo = gr.File(label='Photo to add as layer', file_types=['image'], type='filepath')
                add = gr.Button('Add photo layer')
                with gr.Row():
                    undo = gr.Button('Undo', interactive=False)
                    redo = gr.Button('Redo', interactive=False)
                from retouch.recipes import RECIPES
                recipe = gr.Dropdown(label='Retouch recipe', choices=sorted(RECIPES), value='natural')
                strength = gr.Slider(1, 100, value=100, step=1, label='Result layer strength')
                render = gr.Button('Render recipe into new layer')
                cancel = gr.Button('Cancel recipe render', interactive=False)
                render_status = gr.Markdown('Recipe renderer ready.')
                save_path = gr.Textbox(label='Save project folder', placeholder='/path/to/photo.comp')
                save = gr.Button('Save project')
                export = gr.Button('Export native PNG')
                download = gr.File(label='PNG download', interactive=False)
        status = gr.Markdown('Open a photo or project to start.')
        outputs = [session, preview, layers, name, visible, opacity, mask_enabled,
                   canvas, undo, redo, status, blend, position_x, position_y, flip_x, flip_y,
                   layer_width, layer_height]

        def opening(current, photo_path, path, discard_edits, strokes, kind):
            if current and current.job and not current.job.finalized:
                raise gr.Error('Cancel or finish the recipe render before opening another document.')
            if current and (current.history.dirty or has_strokes(strokes)) and not discard_edits:
                raise gr.Error('Save or select “Discard unsaved edits” before opening another document.')
            try:
                if kind == 'photo':
                    if not photo_path:
                        raise ValueError('Choose a photo first')
                    pixels = _photo(photo_path)
                    doc = Document(pixels.shape[1], pixels.shape[0])
                    doc.add_image(pixels, Path(photo_path).stem)
                else:
                    doc = load_project(path)
                # Check that this subset can be rendered before replacing the session.
                preview_document(doc)
                replacement = EditorSession.create(doc, saved=kind == 'project')
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc
            if current:
                current.close()
            return (*_view(replacement), False)

        for button, kind in ((open_photo, 'photo'), (open_project, 'project')):
            def load(current, p, path, d, strokes, kind=kind):
                return opening(current, p, path, d, strokes, kind)
            button.click(load, [session, photo, project_path, discard, canvas],
                         outputs + [discard], concurrency_id='layer-editor')

        def action(current, selected, strokes, operation, *args):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes) and operation != 'mask':
                raise gr.Error('Apply or erase the current brush strokes first.')
            previous_selection = current.selected_id
            try:
                if operation == 'select':
                    current.history.document.layer(selected)
                    current.selected_id = selected
                else:
                    layer = current.selected()
                    if operation in ('properties', 'transform', 'resize', 'up', 'down', 'remove', 'mask', 'duplicate', 'invert_mask') and layer is None:
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
                                    or not math.isfinite(value) or value != int(value)):
                                raise ValueError('Layer positions must be whole pixel coordinates')
                        current.history.update_layer(layer.id,
                                                     origin=(int(args[0]), int(args[1])),
                                                     flip_x=args[2], flip_y=args[3])
                    elif operation == 'resize':
                        for value in args[:2]:
                            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                                    or not math.isfinite(value) or value != int(value) or value < 1):
                                raise ValueError('Layer dimensions must be positive whole pixels')
                        current.history.update_layer(layer.id, size=(int(args[0]), int(args[1])))
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
                    elif operation == 'mask':
                        current.apply_mask(strokes)
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
                                  (remove, 'remove'), (apply_mask, 'mask'), (add, 'add'),
                                  (undo, 'undo'), (redo, 'redo'), (transform, 'transform'), (resize, 'resize'),
                                  (duplicate, 'duplicate'), (invert_mask, 'invert_mask')):
            def edit(current, selected, strokes, *args, operation=operation):
                return action(current, selected, strokes, operation, *args)
            extra = [name, visible, opacity, mask_enabled, blend] if operation == 'properties' else (
                [position_x, position_y, flip_x, flip_y] if operation == 'transform' else (
                    [layer_width, layer_height] if operation == 'resize' else (
                        [add_photo] if operation == 'add' else [])))
            button.click(edit, [session, layers, canvas] + extra, outputs, concurrency_id='layer-editor')

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

        def exporting(current, strokes):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase brush strokes before exporting')
            path = Path(current.workspace) / (uuid.uuid4().hex + '.png')
            export_png(current.history.document, path)
            return str(path)
        export.click(exporting, [session, canvas], download, concurrency_id='layer-editor')

        from .retouch_jobs import prepare_job, run_job, attach_result

        def prepare_recipe(current, chosen_recipe, chosen_strength, strokes):
            if current is None:
                raise gr.Error('Open a document first')
            if has_strokes(strokes):
                raise gr.Error('Apply or erase brush strokes before rendering')
            if current.job and not current.job.finalized:
                raise gr.Error('A recipe render is already active')
            try:
                current.job = prepare_job(current.history, current.workspace, chosen_recipe, chosen_strength)
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
                    message = 'Recipe result added as a new masked layer.'
                except (ValueError, OSError) as exc:
                    message = str(exc)
            # Keep unapplied strokes on rejection; never silently clear the canvas.
            view = list(_view(current, message))
            if has_strokes(strokes):
                view[7] = gr.update()
            current.job.finalized = True
            return (*view, message, gr.update(interactive=True), gr.update(interactive=False))

        def cancel_recipe(current):
            if current is None or current.job is None:
                return 'No active recipe render.'
            current.job.cancel()
            return 'Cancellation requested; no cancelled result will be attached.'

        render.click(prepare_recipe, [session, recipe, strength, canvas],
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
