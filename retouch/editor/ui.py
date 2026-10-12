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
from .recovery import Autosaver, discard_recovery, list_recoveries, load_recovery

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

    def apply_mask(self, canvas):
        layer = self.selected()
        if layer is None:
            raise ValueError('Select a layer first')
        width, height = ((layer.pixels.shape[1], layer.pixels.shape[0])
                         if layer.pixels is not None else layer.size)
        if width * height > MAX_SURFACE_PIXELS or max(width, height) > 30000:
            raise ValueError('Layer mask exceeds the surface budget')
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
    return (layer.rotation if layer else 0,
            gr.update(choices=folders, value=(layer.parent_id or '') if layer else ''),
            gr.update(choices=sources, value=(layer.mask_source_id or '') if layer else ''),
            gr.update(choices=[(l.name, l.id) for l in doc.ordered_layers()] if doc else [], value=[]),
            selected, settings.get('kind', 'Exposure'), exposure.get('exposure', 0),
            exposure.get('offset', 0), exposure.get('gamma', 1) if settings.get('kind') == 'Exposure' else ranges[0]['gamma'],
            ranges[0]['black'], ranges[0]['white'], ranges[1]['gamma'], ranges[2]['gamma'], ranges[3]['gamma'])


def build_editor_tab():
    """Add a complete independent tab inside the application's Tabs block."""
    with gr.Tab('Layer Editor') as tab:
        session = gr.State(None, delete_callback=lambda value: value.close(keep_recovery=True)
                           if value else None)
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
                    feather = gr.Slider(0, 100, value=0, step=1, label='Feather (canvas pixels)')
                    selection_mode = gr.Radio(['Replace', 'Add', 'Subtract', 'Intersect'], value='Replace', label='Combine selection')
                    create_selection = gr.Button('Create selection')
                    selection_preview = gr.Image(label='Selection coverage (white selected)', interactive=False, image_mode='L')
                    with gr.Row():
                        invert_selection = gr.Button('Invert selection')
                        clear_selection = gr.Button('Clear selection')
                    mask_mode = gr.Radio(['Replace', 'Add', 'Subtract', 'Intersect'], value='Replace', label='Combine with layer mask')
                    mask_selection = gr.Button('Apply selection to selected layer mask')
                    gr.Markdown('Selections use canvas coordinates and remain temporary. '
                                'Apply to a layer or folder mask to save them; this is one undo step.')
                with gr.Accordion('Grading adjustment layers', open=False):
                    grade_kind = gr.Radio(['Exposure', 'Levels', 'Invert'], value='Exposure', label='Adjustment')
                    grade_exposure = gr.Slider(-20, 20, value=0, step=.1, label='Exposure (stops)')
                    grade_offset = gr.Slider(-.5, .5, value=0, step=.01, label='Exposure offset (linear light)')
                    grade_gamma = gr.Slider(.1, 9.99, value=1, step=.01, label='Gamma')
                    grade_black = gr.Slider(0, 254, value=0, step=1, label='Levels black point')
                    grade_white = gr.Slider(1, 255, value=255, step=1, label='Levels white point')
                    grade_red = gr.Slider(.1, 9.99, value=1, step=.01, label='Red gamma')
                    grade_green = gr.Slider(.1, 9.99, value=1, step=.01, label='Green gamma')
                    grade_blue = gr.Slider(.1, 9.99, value=1, step=.01, label='Blue gamma')
                    add_grade = gr.Button('Add adjustment layer')
                    update_grade = gr.Button('Update selected adjustment')
                    gr.Markdown('Adjustments change the layers below and preserve alpha. '
                                'Opacity and masks control coverage. Exposure uses stops, offset and gamma; '
                                'Levels uses black/white points and RGB/channel gamma. Invert has no settings.')
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
                   layer_width, layer_height, rotation, parent_folder, clipping_source, group_members,
                   selection_preview, grade_kind, grade_exposure, grade_offset, grade_gamma, grade_black,
                   grade_white, grade_red, grade_green, grade_blue]

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
            if has_strokes(strokes) and operation != 'mask':
                raise gr.Error('Apply or erase the current brush strokes first.')
            previous_selection = current.selected_id
            try:
                if operation == 'select':
                    current.history.document.layer(selected)
                    current.selected_id = selected
                else:
                    layer = current.selected()
                    if operation in ('properties', 'transform', 'resize', 'fit', 'up', 'down', 'remove', 'mask', 'duplicate', 'invert_mask', 'reset_mask', 'links', 'ungroup', 'selection_mask', 'update_grade') and layer is None:
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
                                                     flip_x=args[2], flip_y=args[3],
                                                     rotation=args[4] if len(args) > 4 else layer.rotation)
                    elif operation == 'links':
                        current.history.update_layer(layer.id, parent_id=args[0] or None,
                                                     mask_source_id=args[1] or None)
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
                    elif operation == 'clear_selection':
                        current.selection = None
                    elif operation == 'invert_selection':
                        doc = current.history.document
                        current.selection = (np.full((doc.height, doc.width), 255, np.uint8)
                                             if current.selection is None else 255 - current.selection)
                    elif operation == 'selection_mask':
                        if current.selection is None:
                            raise ValueError('Create a selection first')
                        current.history.selection_mask(layer.id, current.selection, args[0])
                    elif operation in ('add_grade', 'update_grade'):
                        from .adjustments import adjustment
                        settings = adjustment(*args)
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
                                  (duplicate, 'duplicate'), (invert_mask, 'invert_mask'),
                                  (reset_mask, 'reset_mask'), (fit, 'fit'), (links, 'links'), (group, 'group'),
                                  (ungroup, 'ungroup'), (create_selection, 'selection'),
                                  (clear_selection, 'clear_selection'), (invert_selection, 'invert_selection'),
                                  (mask_selection, 'selection_mask'), (add_grade, 'add_grade'),
                                  (update_grade, 'update_grade')):
            def edit(current, selected, strokes, *args, operation=operation):
                return action(current, selected, strokes, operation, *args)
            extras = {
                'properties': [name, visible, opacity, mask_enabled, blend],
                'transform': [position_x, position_y, flip_x, flip_y, rotation],
                'resize': [layer_width, layer_height, keep_aspect, resize_axis],
                'add': [add_photo], 'links': [parent_folder, clipping_source],
                'group': [group_members],
                'selection': [selection_kind, selection_x, selection_y, selection_width,
                              selection_height, polygon, feather, selection_mode],
                'selection_mask': [mask_mode],
                'add_grade': [grade_kind, grade_exposure, grade_offset, grade_gamma,
                              grade_black, grade_white, grade_red, grade_green, grade_blue],
                'update_grade': [grade_kind, grade_exposure, grade_offset, grade_gamma,
                                 grade_black, grade_white, grade_red, grade_green, grade_blue],
            }
            extra = extras.get(operation, [])
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
