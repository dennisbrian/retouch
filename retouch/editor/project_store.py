"""Validated ``.comp`` reader/writer for the subset the editor core supports.

Follows ``Compositor/IO/ProjectStore.swift`` and ``docs/project-format.md``:
format versions 1-11 are read, version 11 is written, and the upstream checks
(version gating, file naming, path safety, size limits, 8-bit PNGs, grayscale
masks) reject a damaged package before any pixels are decoded. Constructs the
port does not model yet (unsupported adjustment kinds, glow/inner-shadow
effect kinds, unlinked masks, fractional layer sizes and unknown keys) raise
:class:`UnsupportedFeature` instead of being dropped on the next save.

Saving builds the whole package beside the target and swaps it in with
renames, so a failed save leaves the previous package untouched. Layer arrays
are immutable, so an asset whose array is the one last read from or written to
a still-unchanged file is copied instead of re-encoded; the rest are encoded
in parallel at a fast zlib level (a 26 MP layer takes about 2 s instead of 9).

Derived from Compositor (MIT, Copyright (c) 2026 Wonder Assembly LLC), pinned
at 11d8d7a; see ``third_party/compositor/``.
"""
import io
import json
import math
import os
import shutil
import struct
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .document import (
    BLEND_MODES,
    MAX_LAYERS,
    MAX_SIDE,
    MAX_SURFACE_PIXELS,
    SAMPLING,
    Document,
    ColorOverlayEffect,
    InnerGlowEffect,
    InnerShadowEffect,
    Layer,
    LayerEffects,
    OuterGlowEffect,
    ShadowEffect,
    StrokeEffect,
    UnsupportedFeature,
    validate_effects,
)

FORMAT = 'com.compositor.project'
CURRENT_VERSION = 11
SUPPORTED_VERSIONS = range(1, CURRENT_VERSION + 1)
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ASSET_BYTES = 512 * 1024 * 1024

_TOP_KEYS = {'format', 'version', 'colorSpace', 'resolution', 'documentID',
             'width', 'height', 'activeLayerID', 'layers', 'guides'}
_LAYER_KEYS = {'id', 'name', 'isVisible', 'transform', 'imageFile',
               'parentID', 'isGroup', 'opacity', 'blendMode', 'maskFile',
               'maskEnabled', 'maskSourceID', 'adjustment', 'maskPlacement',
               'maskLinked', 'shape', 'effects', 'text'}
_TRANSFORM_KEYS = {'origin', 'size', 'rotation', 'flipX', 'flipY', 'sampling'}
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
# zlib level for project assets. Level 1 encodes a 26 MP RGBA layer about 4x
# faster than Pillow's default 6 for ~20% larger files; flattened exports
# (composite.export_png) keep the default.
ASSET_COMPRESS_LEVEL = 1

# Staging dirs older than this are crash leftovers, not concurrent saves.
_STALE_SAVING_SECONDS = 60 * 60

_ENCODE_WORKERS = max(1, min(4, os.cpu_count() or 1))


class ProjectError(ValueError):
    """The package is not a valid Compositor project (or is over a limit)."""


def _invalid(why):
    return ProjectError('invalid Compositor project: ' + why)


# ---------------------------------------------------------------- reading


def load_project(path):
    """Read a ``.comp`` package into a :class:`Document`."""
    package = Path(path)
    _sweep_interrupted_saves(package)
    manifest, version = _read_manifest(package)
    return _decode(manifest, version, package)


def _read_manifest(package):
    if not package.is_dir() or package.is_symlink():
        raise _invalid('not a package directory')
    data = _read_file(package / 'manifest.json', package, MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _invalid('manifest is not JSON') from exc
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT:
        raise _invalid('not a Compositor manifest')
    version = manifest.get('version')
    if not _is_int(version) or version not in SUPPORTED_VERSIONS:
        raise ProjectError('project format version %r; supported %d-%d'
                           % (version, SUPPORTED_VERSIONS[0], CURRENT_VERSION))
    return manifest, version


def _check_manifest(manifest, version):
    unknown = set(manifest) - _TOP_KEYS
    if unknown:
        raise UnsupportedFeature(f'unknown manifest keys: {sorted(unknown)}')
    if manifest.get('colorSpace') != 'sRGB':
        raise _invalid('colorSpace must be sRGB')
    width, height = manifest.get('width'), manifest.get('height')
    if not (_is_int(width) and _is_int(height)):
        raise _invalid('width/height')
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise ProjectError('canvas exceeds %d px per side' % MAX_SIDE)
    resolution = manifest.get('resolution', 72)
    if not _is_number(resolution) or not 1 <= resolution <= 9600:
        raise _invalid('resolution')
    records = manifest.get('layers')
    if not isinstance(records, list):
        raise _invalid('layers')
    if len(records) > MAX_LAYERS:
        raise ProjectError('more than %d layers' % MAX_LAYERS)
    guides = manifest.get('guides')
    _check_guides(guides, version)

    ids = set()
    parsed = []
    for record in records:
        layer_id, fields = _check_layer(record, version)
        if layer_id in ids:
            raise _invalid('duplicate layer id')
        ids.add(layer_id)
        parsed.append((layer_id, fields))
    active = manifest.get('activeLayerID')
    if active is not None:
        active = _uuid(active)
        if active not in ids:
            raise _invalid('activeLayerID names no layer')
    document_id = _uuid(manifest.get('documentID'))
    surfaces = [fields['size'] for _, fields in parsed if fields['image_file']]
    if any(max(size) > MAX_SIDE for size in surfaces) or sum(
            width * height for width, height in surfaces) > MAX_SURFACE_PIXELS:
        raise ProjectError('rendered layers exceed the size limits')
    # Validate graph dependencies before opening any referenced PNG.
    metadata = [Layer(id=layer_id, **{key: value for key, value in fields.items()
                                     if key not in ('image_file', 'mask_file')})
                for layer_id, fields in parsed]
    try:
        Document(width, height, metadata, document_id=document_id).validate()
    except ValueError as exc:
        raise _invalid(str(exc)) from exc
    return width, height, resolution, guides, active, document_id, parsed


def _decode(manifest, version, package):
    width, height, resolution, guides, active, document_id, parsed = (
        _check_manifest(manifest, version))

    budget = {'image': 0, 'mask': 0}
    layers = []
    for layer_id, fields in parsed:
        pixels = mask = None
        if fields.pop('image_file'):
            pixels = _read_png(package, layer_id + '.png', budget, 'image')
        if fields.pop('mask_file'):
            mask = _read_png(package, layer_id + '.mask.png', budget, 'mask')
        size = fields.pop('size')
        layers.append(Layer(id=layer_id, pixels=pixels, mask=mask, size=size,
                            **fields))
    images = package / 'images'
    for layer in layers:
        _remember(layer.pixels, images / (layer.id + '.png'))
        _remember(layer.mask, images / (layer.id + '.mask.png'))
    document = Document(width=width, height=height, layers=layers,
                        resolution=float(resolution),
                        document_id=document_id,
                        active_layer_id=active, guides=guides)
    document.validate()
    return document


def _check_layer(record, version):
    if not isinstance(record, dict):
        raise _invalid('layer record')
    unknown = set(record) - _LAYER_KEYS
    if unknown:
        raise UnsupportedFeature(f'unknown layer keys: {sorted(unknown)}')
    layer_id = _uuid(record.get('id'))
    name = record.get('name')
    if not isinstance(name, str) or not name.strip() or len(name.encode('utf-8')) > 16_384:
        raise _invalid('layer name')
    visible = record.get('isVisible')
    if not isinstance(visible, bool):
        raise _invalid('isVisible')
    is_group = record.get('isGroup')
    if is_group is not None and not isinstance(is_group, bool):
        raise _invalid('isGroup')
    if version == 1 and (record.get('parentID') is not None or is_group):
        raise _invalid('folders need format version 2')
    opacity = record.get('opacity', 1)
    blend = record.get('blendMode', 'Normal')
    if not _is_number(opacity) or not 0 <= opacity <= 1:
        raise _invalid('opacity')
    if blend not in BLEND_MODES:
        raise _invalid(f'blendMode {blend!r}')
    if version < 3 and (opacity != 1 or blend != 'Normal'):
        raise _invalid('opacity/blend mode need format version 3')
    image_file = record.get('imageFile')
    if image_file is not None and image_file != layer_id + '.png':
        raise _invalid('imageFile must be <layer id>.png')
    mask_file = record.get('maskFile')
    if mask_file is not None and (version < (6 if is_group else 4)
                                  or mask_file != layer_id + '.mask.png'):
        raise _invalid('maskFile must be <layer id>.mask.png (format 4+)')
    mask_enabled = record.get('maskEnabled')
    if mask_enabled is not None and (mask_file is None or not isinstance(mask_enabled, bool)):
        raise _invalid('maskEnabled without a mask')
    if record.get('maskSourceID') is not None and version < 5:
        raise _invalid('clipping masks need format version 5')
    if record.get('adjustment') is not None and version < 7:
        raise _invalid('adjustment layers need format version 7')
    text = record.get('text')
    if text is not None and (not isinstance(text, dict) or image_file is None
                             or (text.get('colorRuns') is not None and version < 10)
                             or (text.get('fontRuns') is not None and version < 11)):
        raise _invalid('text metadata')
    shape = record.get('shape')
    if shape is not None and not isinstance(shape, dict):
        raise _invalid('shape metadata')
    transform = _check_transform(record.get('transform'))

    # Valid upstream, not modelled here yet: refuse rather than drop.
    if is_group and record.get('maskPlacement') is not None:
        # Upstream's export path ignores a folder mask's placement, but the
        # manifest still validates it; the port keeps the same refuse-not-drop
        # rule for the unmodelled folder case.
        raise UnsupportedFeature(f'unlinked folder masks are not supported yet (layer {name!r})')
    if is_group and image_file is not None:
        raise _invalid('folders cannot contain image assets')
    layer_effects = _check_effects(record.get('effects'), name)
    if layer_effects is not None and (image_file is None or is_group):
        raise _invalid('effects need an image layer')
    adjustment = record.get('adjustment')
    if adjustment is not None:
        from .adjustments import validate_adjustment
        validate_adjustment(adjustment)
        if is_group or image_file is not None:
            raise _invalid('invalid adjustment layer')
    if record.get('maskLinked') is False and record.get('maskFile') is None:
        # maskLinked only rides along (it governs interactive moves upstream,
        # not rendering); it still needs a mask to belong to.
        raise _invalid('maskLinked needs a maskFile')
    mask_placement = None
    if record.get('maskPlacement') is not None:
        if mask_file is None:
            raise _invalid('maskPlacement needs a maskFile')
        mask_placement = _check_transform(record.get('maskPlacement'))
    origin, size, rotation = transform['origin'], transform['size'], transform['rotation']
    # Fractional placements are valid upstream (typed values stay exact; only
    # interactive dragging snaps), so they load and round-trip unchanged.
    if any(v != int(v) for v in size):
        raise UnsupportedFeature(f'layer {name!r} has a fractional size')
    return layer_id, {
        'name': name, 'visible': visible, 'opacity': float(opacity), 'blend_mode': blend,
        'origin': tuple(float(v) for v in origin), 'size': tuple(int(v) for v in size),
        'rotation': float(rotation), 'is_group': bool(is_group),
        'parent_id': _uuid(record['parentID']) if record.get('parentID') is not None else None,
        'mask_source_id': _uuid(record['maskSourceID']) if record.get('maskSourceID') is not None else None,
        'adjustment': adjustment,
        'effects': layer_effects,
        'flip_x': transform['flipX'], 'flip_y': transform['flipY'],
        'sampling': transform['sampling'],
        'mask_enabled': True if mask_enabled is None else mask_enabled,
        'mask_placement': mask_placement,
        'mask_linked': record.get('maskLinked') is not False,
        'text': text, 'shape': shape, 'image_file': image_file is not None,
        'mask_file': mask_file is not None}


def _check_transform(transform):
    if not isinstance(transform, dict) or set(transform) != _TRANSFORM_KEYS:
        raise _invalid(f'transform must carry exactly {sorted(_TRANSFORM_KEYS)}')
    origin, size = transform['origin'], transform['size']
    if not (_pair(origin) and _pair(size) and _is_number(transform['rotation'])):
        raise _invalid('transform geometry')
    if not all(1 <= v <= 300_000 for v in size) or any(abs(v) > 1_000_000 for v in origin):
        raise _invalid('transform out of range')
    if not (isinstance(transform['flipX'], bool) and isinstance(transform['flipY'], bool)):
        raise _invalid('transform flips')
    if transform['sampling'] not in SAMPLING:
        raise _invalid('transform sampling')
    return transform


def _check_effects(record, layer_name):
    """Decode upstream's ``effects`` dict, refusing unmodelled effect kinds."""
    if record is None:
        return None
    if not isinstance(record, dict):
        raise _invalid('effects must be a dict')
    for key in record:
        if key not in ('stroke', 'shadow', 'colorOverlay', 'innerShadow', 'outerGlow', 'innerGlow'):
            raise UnsupportedFeature(f'effect kind {key!r} is not supported yet (layer {layer_name!r})')
    stroke = None
    stroke_rec = record.get('stroke')
    if stroke_rec is not None:
        if not isinstance(stroke_rec, dict):
            raise _invalid('stroke must be a dict')
        unknown = set(stroke_rec) - {'enabled', 'size', 'red', 'green', 'blue',
                                     'opacity', 'inside'}
        if unknown:
            raise _invalid(f'unknown stroke keys: {sorted(unknown)}')
        stroke = StrokeEffect(
            size=float(stroke_rec.get('size', 4)),
            red=float(stroke_rec.get('red', 0)),
            green=float(stroke_rec.get('green', 0)),
            blue=float(stroke_rec.get('blue', 0)),
            opacity=float(stroke_rec.get('opacity', 1)),
            inside=stroke_rec.get('inside', False),
            enabled=stroke_rec.get('enabled', True))
    shadow = None
    shadow_rec = record.get('shadow')
    if shadow_rec is not None:
        if not isinstance(shadow_rec, dict):
            raise _invalid('shadow must be a dict')
        unknown = set(shadow_rec) - {'enabled', 'angle', 'distance', 'blur',
                                     'red', 'green', 'blue', 'opacity'}
        if unknown:
            raise _invalid(f'unknown shadow keys: {sorted(unknown)}')
        shadow = ShadowEffect(
            angle=float(shadow_rec.get('angle', 90)),
            distance=float(shadow_rec.get('distance', 20)),
            blur=float(shadow_rec.get('blur', 20)),
            red=float(shadow_rec.get('red', 0)),
            green=float(shadow_rec.get('green', 0)),
            blue=float(shadow_rec.get('blue', 0)),
            opacity=float(shadow_rec.get('opacity', 0.5)),
            enabled=shadow_rec.get('enabled', True))
    overlay = None
    overlay_rec = record.get('colorOverlay')
    if overlay_rec is not None:
        if not isinstance(overlay_rec, dict):
            raise _invalid('colorOverlay must be a dict')
        unknown = set(overlay_rec) - {'enabled', 'red', 'green', 'blue', 'opacity'}
        if unknown:
            raise _invalid(f'unknown colorOverlay keys: {sorted(unknown)}')
        overlay = ColorOverlayEffect(
            red=float(overlay_rec.get('red', 0)),
            green=float(overlay_rec.get('green', 0)),
            blue=float(overlay_rec.get('blue', 0)),
            opacity=float(overlay_rec.get('opacity', 1)),
            enabled=overlay_rec.get('enabled', True))
    inner_shadow = None
    inner_rec = record.get('innerShadow')
    if inner_rec is not None:
        if not isinstance(inner_rec, dict):
            raise _invalid('innerShadow must be a dict')
        unknown = set(inner_rec) - {'enabled', 'angle', 'distance', 'blur',
                                    'red', 'green', 'blue', 'opacity'}
        if unknown:
            raise _invalid(f'unknown innerShadow keys: {sorted(unknown)}')
        inner_shadow = InnerShadowEffect(
            angle=float(inner_rec.get('angle', 90)),
            distance=float(inner_rec.get('distance', 10)),
            blur=float(inner_rec.get('blur', 10)),
            red=float(inner_rec.get('red', 0)),
            green=float(inner_rec.get('green', 0)),
            blue=float(inner_rec.get('blue', 0)),
            opacity=float(inner_rec.get('opacity', 0.5)),
            enabled=inner_rec.get('enabled', True))
    outer_glow = None
    outer_rec = record.get('outerGlow')
    if outer_rec is not None:
        if not isinstance(outer_rec, dict):
            raise _invalid('outerGlow must be a dict')
        unknown = set(outer_rec) - {'enabled', 'size', 'red', 'green', 'blue', 'opacity'}
        if unknown:
            raise _invalid(f'unknown outerGlow keys: {sorted(unknown)}')
        outer_glow = OuterGlowEffect(
            size=float(outer_rec.get('size', 20)),
            red=float(outer_rec.get('red', 1)),
            green=float(outer_rec.get('green', 1)),
            blue=float(outer_rec.get('blue', 1)),
            opacity=float(outer_rec.get('opacity', 0.75)),
            enabled=outer_rec.get('enabled', True))
    inner_glow = None
    inner_glow_rec = record.get('innerGlow')
    if inner_glow_rec is not None:
        if not isinstance(inner_glow_rec, dict):
            raise _invalid('innerGlow must be a dict')
        unknown = set(inner_glow_rec) - {'enabled', 'size', 'red', 'green', 'blue', 'opacity'}
        if unknown:
            raise _invalid(f'unknown innerGlow keys: {sorted(unknown)}')
        inner_glow = InnerGlowEffect(
            size=float(inner_glow_rec.get('size', 10)),
            red=float(inner_glow_rec.get('red', 1)),
            green=float(inner_glow_rec.get('green', 1)),
            blue=float(inner_glow_rec.get('blue', 1)),
            opacity=float(inner_glow_rec.get('opacity', 0.75)),
            enabled=inner_glow_rec.get('enabled', True))
    effects = LayerEffects(stroke=stroke, shadow=shadow, colorOverlay=overlay,
                           innerShadow=inner_shadow, outerGlow=outer_glow,
                           innerGlow=inner_glow)
    try:
        validate_effects(effects)
    except ValueError as exc:
        raise _invalid(str(exc)) from exc
    return effects


def _check_guides(guides, version):
    if guides is None:
        return
    if not isinstance(guides, list):
        raise _invalid('guides')
    if version < 8 and guides:
        raise _invalid('guides need format version 8')
    if len(guides) > 1000:
        raise ProjectError('more than 1000 guides')
    seen = set()
    for guide in guides:
        if not isinstance(guide, dict) or set(guide) != {'id', 'axis', 'position'}:
            raise _invalid('guide record')
        gid = _uuid(guide['id'])
        if gid in seen or guide['axis'] not in ('horizontal', 'vertical') or not (
                _is_number(guide['position']) and abs(guide['position']) <= 1_000_000):
            raise _invalid('guide record')
        seen.add(gid)


def _read_png(package, filename, budget, kind):
    """Decode one embedded PNG after checking its header against the limits."""
    data = _read_file(package / 'images' / filename, package, MAX_ASSET_BYTES)
    width, height, depth, colour = _png_header(data, filename)
    if depth != 8:
        # ProjectStore rejects >8-bit assets; 1/2/4-bit ones never come from
        # Compositor and would be widened silently by Pillow.
        raise ProjectError(f'{filename}: only 8-bit PNGs are supported')
    if kind == 'mask' and colour != 0:
        raise _invalid(f'{filename}: masks must be 8-bit grayscale without alpha')
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE) or (
            budget[kind] + width * height > MAX_SURFACE_PIXELS):
        raise ProjectError(f'{filename} exceeds the size limits')
    budget[kind] += width * height
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None   # bounded above, before decoding
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
                raise _invalid(f'{filename} is not a single-frame PNG')
            image.load()
            if kind == 'mask':
                return np.asarray(image.convert('L'))
            return np.asarray(image.convert('RGBA'))
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(f'{filename} is damaged: {exc}') from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def _png_header(data, filename):
    if len(data) < 33 or data[:8] != _PNG_SIGNATURE or data[12:16] != b'IHDR':
        raise ProjectError(f'{filename} is missing or not a PNG')
    width, height, depth, colour = struct.unpack('>IIBB', data[16:26])
    return width, height, depth, colour


def _read_file(path, package, limit):
    root = os.path.realpath(package) + os.sep
    if not os.path.realpath(path).startswith(root):
        raise _invalid(f'{path.name} escapes the package')
    if path.is_symlink() or not path.is_file():
        raise ProjectError(f'{path.name} is missing')
    if path.stat().st_size > limit:
        raise ProjectError('%s exceeds %d bytes' % (path.name, limit))
    return path.read_bytes()


def _uuid(value):
    try:
        return str(uuid.UUID(value)).upper()
    except (TypeError, ValueError, AttributeError) as exc:
        raise _invalid(f'bad UUID {value!r}') from exc


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _pair(value):
    return isinstance(value, list) and len(value) == 2 and all(map(_is_number, value))


# ---------------------------------------------------------------- writing


def save_project(document, path):
    """Write ``document`` as a version-11 package at ``path`` (``*.comp``).

    An existing package at ``path`` is replaced only after the new one is
    complete; any other existing file or folder there is refused.
    """
    target = Path(path)
    if target.suffix != '.comp':
        raise ValueError('project path must end in .comp')
    if target.exists() or target.is_symlink():
        try:
            existing, version = _read_manifest(target)
            _check_manifest(existing, version)
        except (ValueError, OSError) as exc:
            raise FileExistsError(
                f'{target} exists and is not a supported Compositor project') from exc
    document.validate()
    manifest = _encode(document)
    # Use the reader's schema checks before creating files or retiring a target.
    _check_manifest(manifest, CURRENT_VERSION)
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode('utf-8')
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ProjectError('manifest exceeds 4 MiB')

    token = uuid.uuid4().hex[:12]
    staging = target.with_name(f'.{target.name}.saving-{token}')
    staging.mkdir()
    try:
        images = staging / 'images'
        images.mkdir()
        assets = _assets(document)
        _write_assets(assets, images)
        (staging / 'manifest.json').write_bytes(payload)
        _swap_in(staging, target, token)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name, array, _ in assets:
        _remember(array, target / 'images' / name)
    return target


def _assets(document):
    assets = []
    for layer in document.layers:
        if layer.pixels is not None:
            assets.append((layer.id + '.png', layer.pixels, 'RGBA'))
        if layer.mask is not None:
            assets.append((layer.id + '.mask.png', layer.mask, 'L'))
    return assets


def _write_assets(assets, images):
    """Copy assets whose bytes are already on disk; encode the rest."""
    pending = []
    for name, array, mode in assets:
        source = _known_file(array)
        if source is not None:
            try:
                shutil.copyfile(source, images / name)
                continue
            except OSError:
                pass
        pending.append((name, array, mode))
    if not pending:
        return

    def encode(item):
        name, array, mode = item
        Image.fromarray(array, mode).save(
            images / name, format='PNG', compress_level=ASSET_COMPRESS_LEVEL)

    if len(pending) == 1:
        encode(pending[0])
        return
    with ThreadPoolExecutor(min(_ENCODE_WORKERS, len(pending))) as pool:
        for _ in pool.map(encode, pending):   # re-raises the first failure
            pass


# Immutable array -> the PNG holding exactly its bytes, while that file is
# untouched since it was read or written. Keyed by id() and checked against a
# weak reference, so a recycled id never matches and arrays are not kept alive.
_KNOWN = {}
_KNOWN_LOCK = threading.Lock()


def _signature(path):
    stat = os.stat(path)
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _remember(array, path):
    if array is None:
        return
    key = id(array)
    try:
        signature = _signature(path)
        ref = weakref.ref(array, lambda _: _forget(key))
    except (OSError, TypeError):
        return
    with _KNOWN_LOCK:
        _KNOWN[key] = (ref, os.path.realpath(path), signature)


def _forget(key):
    with _KNOWN_LOCK:
        entry = _KNOWN.get(key)
        if entry is not None and entry[0]() is None:
            del _KNOWN[key]


def _known_file(array):
    with _KNOWN_LOCK:
        entry = _KNOWN.get(id(array))
    if entry is None or entry[0]() is not array or array.flags.writeable:
        return None
    _, path, signature = entry
    try:
        if _signature(path) != signature:
            return None
    except OSError:
        return None
    return path


def _swap_in(staging, target, token):
    if not target.exists():
        os.rename(staging, target)
        return
    retired = target.with_name(f'.{target.name}.replaced-{token}')
    os.rename(target, retired)
    try:
        os.rename(staging, target)
    except BaseException:
        os.rename(retired, target)
        raise
    shutil.rmtree(retired, ignore_errors=True)


def _sweep_interrupted_saves(target):
    """Recover a package stranded by a crash between ``_swap_in``'s renames.

    A process killed after the old package was retired but before the new one
    was swapped in leaves the target missing and the only surviving copy under
    ``.<name>.replaced-<token>``. Restore the newest such sibling (the old
    package — a complete, valid state) and drop stale staging dirs. Old-or-new
    then holds across process crashes, not just exceptions.
    """
    parent = target.parent
    if not parent.is_dir():
        return
    best = None
    for entry in parent.iterdir():
        if entry.name.startswith(f'.{target.name}.replaced-'):
            if best is None or entry.stat().st_mtime > best.stat().st_mtime:
                if best is not None:
                    shutil.rmtree(best, ignore_errors=True)
                best = entry
        elif entry.name.startswith(f'.{target.name}.saving-'):
            # Only crash leftovers: a concurrent writer's staging dir matches
            # this prefix for its whole write window, so sweeping by name
            # alone would delete a live save out from under it. Age-gate
            # instead — no legitimate save takes an hour.
            try:
                if entry.stat().st_mtime < time.time() - _STALE_SAVING_SECONDS:
                    shutil.rmtree(entry, ignore_errors=True)
            except FileNotFoundError:
                pass
    if best is not None and not target.exists():
        os.rename(best, target)
    elif best is not None:
        shutil.rmtree(best, ignore_errors=True)


def _encode(document):
    manifest = {
        'format': FORMAT, 'version': CURRENT_VERSION, 'colorSpace': 'sRGB',
        'documentID': document.document_id, 'width': document.width,
        'height': document.height, 'resolution': document.resolution,
        'layers': [_encode_layer(layer) for layer in document.layers],
    }
    if document.active_layer_id is not None:
        manifest['activeLayerID'] = document.active_layer_id
    if document.guides is not None:
        manifest['guides'] = document.guides
    return manifest


def _encode_layer(layer):
    record = {
        'id': layer.id, 'name': layer.name, 'isVisible': layer.visible,
        'isGroup': layer.is_group, 'opacity': layer.opacity, 'blendMode': layer.blend_mode,
        'transform': {'origin': list(layer.origin), 'size': list(layer.size),
                      'rotation': layer.rotation, 'flipX': layer.flip_x, 'flipY': layer.flip_y,
                      'sampling': layer.sampling},
    }
    if layer.parent_id is not None:
        record['parentID'] = layer.parent_id
    if layer.mask_source_id is not None:
        record['maskSourceID'] = layer.mask_source_id
    if layer.adjustment is not None:
        record['adjustment'] = layer.adjustment
    if layer.pixels is not None:
        record['imageFile'] = layer.id + '.png'
    if layer.mask is not None:
        record['maskFile'] = layer.id + '.mask.png'
        record['maskEnabled'] = layer.mask_enabled
        if layer.mask_placement is not None:
            record['maskPlacement'] = {
                'origin': list(layer.mask_placement['origin']),
                'size': list(layer.mask_placement['size']),
                'rotation': float(layer.mask_placement['rotation']),
                'flipX': bool(layer.mask_placement['flipX']),
                'flipY': bool(layer.mask_placement['flipY']),
                'sampling': layer.mask_placement['sampling']}
        if not layer.mask_linked:
            record['maskLinked'] = False
    if layer.text is not None:
        record['text'] = layer.text
    if layer.shape is not None:
        record['shape'] = layer.shape
    if layer.effects is not None and not layer.effects.is_empty():
        record['effects'] = _encode_effects(layer.effects)
    return record


def _encode_effects(effects):
    """Encode a :class:`LayerEffects` record matching upstream's schema."""
    out = {}
    if effects.stroke is not None:
        s = effects.stroke
        rec = {'size': s.size, 'red': s.red, 'green': s.green, 'blue': s.blue,
               'opacity': s.opacity, 'inside': s.inside}
        if not s.enabled:
            rec['enabled'] = False
        out['stroke'] = rec
    if effects.shadow is not None:
        s = effects.shadow
        rec = {'angle': s.angle, 'distance': s.distance, 'blur': s.blur,
               'red': s.red, 'green': s.green, 'blue': s.blue, 'opacity': s.opacity}
        if not s.enabled:
            rec['enabled'] = False
        out['shadow'] = rec
    if effects.colorOverlay is not None:
        o = effects.colorOverlay
        rec = {'red': o.red, 'green': o.green, 'blue': o.blue, 'opacity': o.opacity}
        if not o.enabled:
            rec['enabled'] = False
        out['colorOverlay'] = rec
    if effects.innerShadow is not None:
        s = effects.innerShadow
        rec = {'angle': s.angle, 'distance': s.distance, 'blur': s.blur,
               'red': s.red, 'green': s.green, 'blue': s.blue, 'opacity': s.opacity}
        if not s.enabled:
            rec['enabled'] = False
        out['innerShadow'] = rec
    for key, glow in (('outerGlow', effects.outerGlow), ('innerGlow', effects.innerGlow)):
        if glow is not None:
            rec = {'size': glow.size, 'red': glow.red, 'green': glow.green,
                   'blue': glow.blue, 'opacity': glow.opacity}
            if not glow.enabled:
                rec['enabled'] = False
            out[key] = rec
    return out
